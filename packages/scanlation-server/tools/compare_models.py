"""Compare candidate DETECTION and Japanese-OCR models on the same manga page.

This is a RESEARCH harness, deliberately separate from the plugin system: each
candidate model is wrapped in a small adapter that lazy-imports its framework and
is *skipped with a clear reason* if its deps/weights aren't installed. So you can
enable models one at a time (pip install ..., pull an ollama tag) and re-run.

  * detect : every available detector draws its boxes on the page; the panels are
             tiled into ONE side-by-side montage (boxes + count + ms per model).
  * ocr    : one reference detector makes the deskewed crops, then every available
             OCR model reads the SAME crops; the text is printed aligned per crop.

    ../../venv/Scripts/python tools/compare_models.py list
    ../../venv/Scripts/python tools/compare_models.py detect page.png [--only ogkalu_rtdetr,kiuyha_yolo]   # -> compare_out/
    ../../venv/Scripts/python tools/compare_models.py ocr page.png [--ref-detector ogkalu_rtdetr] [--device both] [--only mangaocr,qwen3vl]

Once a winner is clear, promote just that model to a real scanlation-<name>
plugin (EngineBase + entry_points). This script stays throwaway research tooling,
so heavy frameworks (ultralytics, transformers VLMs) are NOT project deps — they
import lazily and only when their adapter actually runs.
"""
from __future__ import annotations

import argparse
import base64
import io
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import _bootstrap  # noqa: F401 - side effects: add package root to sys.path, UTF-8 stdio

from PIL import Image, ImageDraw, ImageFont

# OCR prompt shared by every VLM OCR adapter (ollama + transformers). Kept terse
# so general VLMs don't narrate; manga-ocr ignores it (it takes only the image).
OCR_PROMPT = (
    "You are an OCR engine. Read ALL the Japanese text in this manga panel, "
    "top-to-bottom then right-to-left. Output ONLY the transcribed text, no "
    "explanation, no romaji, no translation."
)

# distinct colors per detector class label (RGB); unknown labels cycle a palette
_PALETTE = [(255, 40, 40), (40, 140, 255), (40, 200, 90), (255, 170, 0),
            (200, 60, 220), (0, 200, 200), (255, 90, 160), (150, 110, 60)]


# --------------------------------------------------------------------------- #
# normalized result types
# --------------------------------------------------------------------------- #
@dataclass
class Box:
    xyxy: tuple[float, float, float, float]
    label: str = ""
    score: float = 0.0
    polygon: list | None = None  # rotated quad / seg contour, else None (axis-aligned)


@dataclass
class DetResult:
    boxes: list[Box] = field(default_factory=list)
    ms: float = 0.0


# --------------------------------------------------------------------------- #
# adapter base
# --------------------------------------------------------------------------- #
class Adapter:
    id: str = ""
    kind: str = ""          # "detect" | "ocr"
    label: str = ""
    install_hint: str = ""
    unverified: bool = False  # repo id / api not yet confirmed -> reported, not run
    heavy: bool = False       # multi-GB download -> only runs when named in --only
    device: str | None = None       # OCR only: "cpu"/"cuda"/None(auto); set per run by cmd_ocr
    device_switchable: bool = True   # False = the device is out of our hands (ollama's server)
    cpu_ok: bool = True              # False = too heavy for CPU (e.g. 7B) -> skip the cpu pass

    def available(self) -> tuple[bool, str]:
        """(ok, reason). Cheap: import checks + weight presence, never a full load."""
        raise NotImplementedError

    def load(self) -> None:
        pass


def _imports_ok(*mods: str) -> tuple[bool, str]:
    import importlib.util
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    return (not missing, "" if not missing else f"missing: {', '.join(missing)}")


def _hf_weight_file(repo: str, exts=(".pt", ".onnx")) -> str:
    """First weight file in an HF repo (so we don't hardcode a filename that the
    author might rename). Raises if the repo has none of the extensions."""
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files(repo)
    for ext in exts:
        for f in sorted(files):
            if f.endswith(ext):
                return hf_hub_download(repo, f)
    raise RuntimeError(f"{repo}: no {exts} weight file found (has: {files[:8]}...)")


def _torch_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _resolve_device(device: str | None) -> str:
    """None/'auto' -> best available ('cuda' if present else 'cpu'); else pass through."""
    return _torch_device() if device in (None, "auto") else device


def _iou(a: tuple, b: tuple) -> float:
    """Intersection over union of two xyxy boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _ios(a: tuple, b: tuple) -> float:
    """Intersection over the *smaller* box's area — catches a small box nested
    inside a big one (their IoU is low, but IoS is high)."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def dedup_boxes(boxes: list[Box], nms_iou: float | None, contain_thresh: float | None) -> list[Box]:
    """Greedy suppression, highest score first: drop a box if it overlaps an
    already-kept (higher-score) box past ``nms_iou`` (IoU, same-size duplicates)
    OR is ``contain_thresh`` inside/around one (IoS, nested small-in-big). Either
    threshold None disables that half. Set a threshold to 1.0 to effectively off."""
    if not boxes or (nms_iou is None and contain_thresh is None):
        return boxes
    kept: list[Box] = []
    for b in sorted(boxes, key=lambda x: -x.score):
        if any(
            (nms_iou is not None and _iou(b.xyxy, k.xyxy) >= nms_iou)
            or (contain_thresh is not None and _ios(b.xyxy, k.xyxy) >= contain_thresh)
            for k in kept
        ):
            continue
        kept.append(b)
    return kept


# --------------------------------------------------------------------------- #
# DETECTION adapters
# --------------------------------------------------------------------------- #
class UltralyticsDetAdapter(Adapter):
    """Any ultralytics YOLO(.pt) detector/segmenter from an HF repo. Uniform API:
    weight auto-discovered, YOLO(path), model(img). seg models also yield masks."""
    kind = "detect"

    def __init__(self, id_: str, repo: str, label: str, *, seg: bool = False,
                 conf: float = 0.3, imgsz: int = 1024):
        self.id, self.repo, self.label, self.seg, self.conf, self.imgsz = id_, repo, label, seg, conf, imgsz
        self.install_hint = f"pip install ultralytics huggingface_hub   # repo {repo}"

    def available(self) -> tuple[bool, str]:
        return _imports_ok("ultralytics", "huggingface_hub")

    def load(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(_hf_weight_file(self.repo, (".pt",)))

    def detect(self, img: Image.Image) -> list[Box]:
        res = self._model.predict(img, verbose=False, conf=self.conf, imgsz=self.imgsz)[0]
        names = res.names or {}
        out: list[Box] = []
        if self.seg and getattr(res, "masks", None) is not None and res.masks is not None:
            classes = res.boxes.cls.tolist() if res.boxes is not None else []
            confs = res.boxes.conf.tolist() if res.boxes is not None else []
            for i, poly in enumerate(res.masks.xy):
                pts = [(float(x), float(y)) for x, y in poly]
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                cid = int(classes[i]) if i < len(classes) else 0
                sc = float(confs[i]) if i < len(confs) else 0.0
                out.append(Box((min(xs), min(ys), max(xs), max(ys)), names.get(cid, str(cid)), sc, pts))
            return out
        if res.boxes is not None:
            for b in res.boxes:
                x0, y0, x1, y1 = b.xyxy[0].tolist()
                out.append(Box((x0, y0, x1, y1), names.get(int(b.cls[0]), str(int(b.cls[0]))), float(b.conf[0])))
        return out


class TransformersDetAdapter(Adapter):
    """A transformers object-detection model (e.g. RT-DETR-v2) via the Auto* API,
    so the exact model class is picked from the repo config — no hardcoded class."""
    kind = "detect"

    def __init__(self, id_: str, repo: str, label: str, *, conf: float = 0.3,
                 keep_labels: set[str] | None = None, nms_iou: float | None = None,
                 contain_thresh: float | None = None, unverified: bool = False):
        self.id, self.repo, self.label, self.conf = id_, repo, label, conf
        self.keep_labels, self.unverified = keep_labels, unverified
        self.nms_iou, self.contain_thresh = nms_iou, contain_thresh  # post-process dedup (tunable via --opt)
        self.install_hint = f"pip install transformers torch   # repo {repo}"

    def available(self) -> tuple[bool, str]:
        if self.unverified:
            return False, "repo/format unverified — confirm then set unverified=False"
        return _imports_ok("transformers", "torch")

    def load(self) -> None:
        import torch
        from transformers import AutoImageProcessor
        self._dev = _torch_device()
        self._proc = AutoImageProcessor.from_pretrained(self.repo)
        try:  # Auto covers most detection models
            from transformers import AutoModelForObjectDetection
            model = AutoModelForObjectDetection.from_pretrained(self.repo)
        except (ValueError, KeyError):  # rt_detr_v2 on older transformers: name the class
            from transformers import RTDetrV2ForObjectDetection
            model = RTDetrV2ForObjectDetection.from_pretrained(self.repo)
        self._model = model.to(self._dev).eval()
        self._torch = torch

    def detect(self, img: Image.Image) -> list[Box]:
        inputs = self._proc(images=img, return_tensors="pt").to(self._dev)
        with self._torch.no_grad():
            out = self._model(**inputs)
        sizes = self._torch.tensor([img.size[::-1]]).to(self._dev)  # (h, w)
        r = self._proc.post_process_object_detection(out, target_sizes=sizes, threshold=self.conf)[0]
        id2label = getattr(self._model.config, "id2label", {})
        boxes = [
            Box(tuple(box.tolist()), id2label.get(int(lab), str(int(lab))), float(sc))
            for sc, lab, box in zip(r["scores"], r["labels"], r["boxes"])
        ]
        if self.keep_labels is not None:  # drop the whole-bubble container, keep text regions
            boxes = [b for b in boxes if b.label in self.keep_labels]
        return dedup_boxes(boxes, self.nms_iou, self.contain_thresh)


# --------------------------------------------------------------------------- #
# OCR adapters
# --------------------------------------------------------------------------- #
class MangaOcrAdapter(Adapter):
    id, kind, label = "mangaocr", "ocr", "manga-ocr"
    install_hint = "pip install manga-ocr   (or: -e packages/scanlation-mangaocr)"

    def available(self) -> tuple[bool, str]:
        return _imports_ok("manga_ocr")

    def load(self) -> None:
        from manga_ocr import MangaOcr
        self._m = MangaOcr(force_cpu=_resolve_device(self.device) == "cpu")

    def recognize(self, crop: Image.Image) -> str:
        return self._m(crop).strip()


class OllamaVlmAdapter(Adapter):
    """OCR via a vision model served by ollama (POST /api/generate with images)."""
    kind = "ocr"
    device_switchable = False  # the ollama server decides CPU/GPU, not this client

    def __init__(self, id_: str, tag: str, label: str, *, unverified: bool = False):
        self.id, self.tag, self.label, self.unverified = id_, tag, label, unverified
        self.endpoint = os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434/api").rstrip("/")
        self.install_hint = f"ollama pull {tag}   (endpoint OLLAMA_ENDPOINT={self.endpoint})"

    def available(self) -> tuple[bool, str]:
        if self.unverified:
            return False, "ollama tag unverified — confirm the model exists on ollama"
        ok, why = _imports_ok("httpx")
        if not ok:
            return False, why
        try:  # is the tag actually pulled?
            import httpx
            tags = httpx.get(f"{self.endpoint}/tags", timeout=3.0).json().get("models", [])
            names = {m.get("name", "").split(":")[0] for m in tags}
            if self.tag.split(":")[0] not in names:
                return False, f"tag '{self.tag}' not pulled (ollama pull {self.tag})"
            return True, ""
        except Exception as exc:  # noqa: BLE001
            return False, f"ollama unreachable at {self.endpoint} ({exc})"

    def load(self) -> None:
        import httpx
        self._client = httpx.Client(timeout=120.0)

    def recognize(self, crop: Image.Image) -> str:
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        body = {"model": self.tag, "prompt": OCR_PROMPT, "images": [b64], "stream": False, "think": False}
        r = self._client.post(f"{self.endpoint}/generate", json=body)
        r.raise_for_status()
        return (r.json().get("response") or "").strip()


class HfVlmAdapter(Adapter):
    """Generic transformers image-text-to-text VLM used as an OCR engine
    (MangaLMM, dots.ocr, Qwen*-VL local, ...). Chat template + OCR_PROMPT."""
    kind = "ocr"

    def __init__(self, id_: str, repo: str, label: str, *, trust_remote_code: bool = False,
                 prompt: str = OCR_PROMPT, max_new_tokens: int = 256,
                 unverified: bool = False, heavy: bool = False, cpu_ok: bool = True,
                 processor_repo: str | None = None):
        self.id, self.repo, self.label = id_, repo, label
        self.trust, self.prompt, self.max_new_tokens = trust_remote_code, prompt, max_new_tokens
        self.unverified, self.heavy, self.cpu_ok = unverified, heavy, cpu_ok
        self.processor_repo = processor_repo  # load the processor from a base repo when the
        self.install_hint = f"pip install transformers torch accelerate   # repo {repo}"  # finetune's own is unreadable

    def available(self) -> tuple[bool, str]:
        if self.unverified:
            return False, "repo id / load path unverified — confirm then enable"
        return _imports_ok("transformers", "torch")

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        dev = _resolve_device(self.device)  # "cpu"/"cuda"; fp32 on CPU (fp16 there is a trap)
        self._proc = AutoProcessor.from_pretrained(self.processor_repo or self.repo, trust_remote_code=self.trust)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.repo, trust_remote_code=self.trust,
            torch_dtype=(torch.float32 if dev == "cpu" else "auto"), device_map=dev,
        ).eval()

    def recognize(self, crop: Image.Image) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}]
        text = self._proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self._proc(text=[text], images=[crop], return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._proc.decode(gen, skip_special_tokens=True).strip()


class HfCausalVlmAdapter(Adapter):
    """VLMs exposed through the remote-code AutoModelForCausalLM path rather than
    AutoModelForImageTextToText (dots.ocr, PaddleOCR-VL). Same chat-template call,
    a per-model literal OCR prompt, and extra import gates (e.g. qwen_vl_utils)."""
    kind = "ocr"

    def __init__(self, id_: str, repo: str, label: str, prompt: str, *,
                 needs: tuple[str, ...] = ("transformers", "torch"),
                 max_new_tokens: int = 1024, heavy: bool = True):
        self.id, self.repo, self.label, self.prompt = id_, repo, label, prompt
        self.needs, self.max_new_tokens, self.heavy = needs, max_new_tokens, heavy
        self.install_hint = f"pip install {' '.join(n for n in needs if n != 'torch')} torch accelerate   # repo {repo}"

    def available(self) -> tuple[bool, str]:
        return _imports_ok(*self.needs)

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        dev = _resolve_device(self.device)  # "cpu"/"cuda"; fp32 on CPU (fp16 there is a trap)
        src = self.repo
        if "." in self.repo.split("/")[-1]:  # a '.' in the repo name breaks the dynamic-module
            from huggingface_hub import snapshot_download  # loader -> load from the (hash-named) local snapshot
            src = snapshot_download(self.repo)
        self._proc = AutoProcessor.from_pretrained(src, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            src, trust_remote_code=True,
            torch_dtype=(torch.float32 if dev == "cpu" else "auto"), device_map=dev,
        ).eval()

    def recognize(self, crop: Image.Image) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": crop},
                                                 {"type": "text", "text": self.prompt}]}]
        text = self._proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._proc(text=[text], images=[crop], return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                   repetition_penalty=1.1)  # dots.ocr loops on some crops under greedy
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._proc.decode(gen, skip_special_tokens=True).strip()


class MitOcrAdapter(Adapter):
    """manga-image-translator's 48px / 48px_ctc line OCR (vendored:
    tools/vendored/_mit_ocr*.py + tools/vendored/_mit_weights/). Those models read
    ONE 48px text line, so the runner splits each bubble crop into columns/rows
    first (ink-projection), OCRs each, and joins in reading order."""
    kind = "ocr"

    def __init__(self, id_: str, variant: str, label: str):
        self.id, self.variant, self.label = id_, variant, label
        self.install_hint = ("vendored — weights auto-included in tools/vendored/_mit_weights/ "
                             "(from zyddnys/manga-image-translator release beta-0.3)")

    def _ckpt(self) -> str:
        return "ocr-ctc.ckpt" if self.variant == "ctc" else "ocr_ar_48px.ckpt"

    def available(self) -> tuple[bool, str]:
        ok, why = _imports_ok("torch", "cv2")
        if not ok:
            return False, why
        wpath = os.path.join(os.path.dirname(__file__), "vendored", "_mit_weights", self._ckpt())
        return (True, "") if os.path.exists(wpath) else (False, f"missing weights {self._ckpt()} in tools/vendored/_mit_weights/")

    def load(self) -> None:
        from vendored._mit_ocr import MitOCR
        self._m = MitOCR(self.variant)
        self._m.load(_resolve_device(self.device))

    def recognize(self, crop: Image.Image) -> str:
        import numpy as np
        return self._m.recognize(np.array(crop.convert("RGB")))


# --------------------------------------------------------------------------- #
# adapter registry  (repo ids / tags / classes verified 2026-07 against HF cards
# + ollama library; see the research notes. `heavy` VLMs are multi-GB downloads
# so they only run when named in --only, never by accident on a bare `ocr` run.)
# --------------------------------------------------------------------------- #
def all_adapters() -> list[Adapter]:
    return [
        # --- detectors ---
        TransformersDetAdapter("ogkalu_rtdetr", "ogkalu/comic-text-and-bubble-detector",
                               "ogkalu RT-DETR-v2 (text_bubble+text_free)",
                               keep_labels={"text_bubble", "text_free"},
                               conf=0.6, nms_iou=0.6, contain_thresh=0.85),  # conf=0.6 = tuned sweet spot
        UltralyticsDetAdapter("kiuyha_yolo", "Kiuyha/Manga-Bubble-YOLO",
                              "Kiuyha Manga-Bubble-YOLO26 (needs recent ultralytics)", imgsz=1280),
        UltralyticsDetAdapter("ogkalu_yolov8m", "ogkalu/comic-speech-bubble-detector-yolov8m",
                              "ogkalu speech-bubble YOLOv8m", imgsz=1024),
        UltralyticsDetAdapter("kitsumed_seg", "kitsumed/yolov8m_seg-speech-bubble",
                              "kitsumed YOLOv8m-seg (mask)", seg=True, imgsz=1024),
        # --- ocr ---
        MangaOcrAdapter(),
        OllamaVlmAdapter("qwen3vl", "qwen3-vl:4b-instruct", "Qwen3-VL 4B Instruct (ollama, non-thinking)"),
        HfCausalVlmAdapter("dots_ocr", "rednote-hilab/dots.ocr", "dots.ocr",
                           "Extract the text content from this image.",
                           needs=("transformers", "torch", "qwen_vl_utils")),
        HfVlmAdapter("paddleocr_vl", "PaddlePaddle/PaddleOCR-VL", "PaddleOCR-VL (0.9B, native)",
                     prompt="OCR:", max_new_tokens=1024),  # native transformers 5.x impl (remote code is upstream-broken)
        HfVlmAdapter("paddleocr_manga", "jzhang533/PaddleOCR-VL-For-Manga",
                     "PaddleOCR-VL manga fine-tune (jzhang533)", prompt="OCR:", max_new_tokens=1024,
                     processor_repo="PaddlePaddle/PaddleOCR-VL"),  # native weights + base processor (its own is 4.x-format)
        MitOcrAdapter("mit_48px_ctc", "ctc", "48px_ctc (manga-image-translator, CTC)"),
        MitOcrAdapter("mit_48px", "48px", "48px (manga-image-translator, attention)"),
    ]


def _select(kind: str, only: str | None, exclude: str | None = None) -> list[Adapter]:
    ads = [a for a in all_adapters() if a.kind == kind]
    if only:
        want = {s.strip() for s in only.split(",")}
        ads = [a for a in ads if a.id in want]
    if exclude:
        drop = {s.strip() for s in exclude.split(",")}
        ads = [a for a in ads if a.id not in drop]
    return ads


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:  # noqa: BLE001
            pass
    return ImageFont.load_default()


def _color(label: str, seen: dict[str, tuple]) -> tuple:
    if label not in seen:
        seen[label] = _PALETTE[len(seen) % len(_PALETTE)]
    return seen[label]


def render_panel(img: Image.Image, res: DetResult, title: str, *, by_class: bool = False) -> Image.Image:
    """Draw a detector's boxes on the page with a title strip above. Boxes are a
    uniform red by default (matching tools/visualize.py / the p8_after sample);
    pass by_class to color each class distinctly (e.g. rtdetr's
    bubble/text_bubble/text_free)."""
    canvas = img.convert("RGB").copy()
    d = ImageDraw.Draw(canvas)
    seen: dict[str, tuple] = {}
    counts: dict[str, int] = {}
    for b in res.boxes:
        counts[b.label] = counts.get(b.label, 0) + 1
        col = _color(b.label, seen) if by_class else (255, 0, 0)
        if b.polygon and len(b.polygon) >= 3:
            d.polygon([(float(x), float(y)) for x, y in b.polygon], outline=col, width=3)
        else:
            d.rectangle(b.xyxy, outline=col, width=3)
    strip_h = 34
    out = Image.new("RGB", (canvas.width, canvas.height + strip_h), (20, 20, 20))
    out.paste(canvas, (0, strip_h))
    dd = ImageDraw.Draw(out)
    legend = "  ".join(f"{k}={n}" for k, n in counts.items()) or "0"
    ms = f"  {res.ms:.0f}ms" if res.ms else ""  # omit a meaningless 0ms (e.g. the ba mode)
    dd.text((6, 7), f"{title}   {len(res.boxes)} boxes{ms}   [{legend}]",
            fill=(255, 255, 255), font=_font(20))
    return out


def montage(panels: list[Image.Image], out_path: Path, width: int = 720) -> None:
    if not panels:
        return
    scaled = [p.resize((width, int(p.height * width / p.width))) for p in panels]
    ncols = min(len(scaled), 2 if len(scaled) <= 4 else 3)
    nrows = math.ceil(len(scaled) / ncols)
    cw, ch, gap = width, max(p.height for p in scaled), 10
    grid = Image.new("RGB", (ncols * cw + (ncols + 1) * gap, nrows * ch + (nrows + 1) * gap), (60, 60, 60))
    for i, p in enumerate(scaled):
        r, c = divmod(i, ncols)
        grid.paste(p, (gap + c * (cw + gap), gap + r * (ch + gap)))
    grid.save(out_path)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def _explicit_ids(only: str | None) -> set[str]:
    return {s.strip() for s in only.split(",")} if only else set()


def _coerce(cur, v: str):
    """Coerce a --opt string value to the type of the attribute it overrides."""
    if isinstance(cur, bool):
        return v.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int):
        return int(v)
    if isinstance(cur, float):
        return float(v)
    if cur is None:  # an off-by-default knob (e.g. nms_iou): accept a number
        try:
            return float(v)
        except ValueError:
            return v
    return v


def _apply_opts(adapters: list[Adapter], opts: list[str] | None) -> None:
    """Set tunable attributes on adapters from KEY=VALUE strings (e.g.
    conf=0.5 nms_iou=0.5). An adapter without the attribute silently ignores it,
    so `--opt conf=0.5` only affects the detectors that actually have a conf."""
    pairs = {}
    for o in opts or []:
        if "=" not in o:
            sys.exit(f"--opt expects KEY=VALUE, got {o!r}")
        k, v = o.split("=", 1)
        pairs[k.strip()] = v.strip()
    for a in adapters:
        for k, v in pairs.items():
            if hasattr(a, k):
                setattr(a, k, _coerce(getattr(a, k), v))


def _run_available(adapters: list[Adapter], explicit: set[str]):
    """Yield (adapter, ok, reason). Prints a live skip/run line per adapter. A
    `heavy` adapter is skipped unless the user named it in --only, so a bare run
    never kicks off a multi-GB model download by surprise."""
    for a in adapters:
        if a.heavy and a.id not in explicit:
            ok, why = False, "heavy (multi-GB) — add to --only to run"
        else:
            ok, why = a.available()
        status = "run " if ok else "SKIP"
        print(f"  [{status}] {a.id:<14} {a.label}" + ("" if ok else f"   -- {why}"), file=sys.stderr)
        yield a, ok, why


def cmd_list(_args) -> None:
    print("adapters (available on this machine = [run]):\n", file=sys.stderr)
    for a in all_adapters():
        ok, why = a.available()
        tag = "run " if ok else "SKIP"
        heavy = "  (heavy: name it in --only)" if a.heavy else ""
        print(f"  [{tag}] {a.kind:<7} {a.id:<14} {a.label}{heavy}")
        if not ok:
            print(f"           -> {why}\n           install: {a.install_hint}")


def cmd_detect(args) -> None:
    img = Image.open(args.image).convert("RGB")
    panel_dir = Path(args.panels)
    panel_dir.mkdir(parents=True, exist_ok=True)
    ads = _select("detect", args.only, args.exclude)
    _apply_opts(ads, args.opt)
    panels: list[Image.Image] = []
    for a, ok, _ in _run_available(ads, _explicit_ids(args.only)):
        if not ok:
            continue
        try:
            a.load()
            t0 = time.perf_counter()
            boxes = a.detect(img)
            res = DetResult(boxes, (time.perf_counter() - t0) * 1000)
            print(f"    {a.id}: {len(boxes)} boxes in {res.ms:.0f}ms", file=sys.stderr)
            panel = render_panel(img, res, a.label, by_class=args.by_class)
            panel.save(panel_dir / f"{a.id}.png")  # full-res, for zooming in
            panels.append(panel)
        except Exception as exc:  # noqa: BLE001 - one broken model must not kill the run
            print(f"    {a.id}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    montage(panels, out)
    print(f"\nwrote {out}  +  {len(panels)} full-res panel(s) -> {panel_dir}/", file=sys.stderr)


def _load_ref_detector(ref_id: str) -> Adapter:
    """Load the reference detector (the decided box model) once, so a batch can
    reuse it across every image instead of reloading it per image."""
    ad = next((a for a in all_adapters() if a.id == ref_id and a.kind == "detect"), None)
    if ad is None:
        sys.exit(f"unknown ref detector: {ref_id}")
    ok, why = ad.available()
    if not ok:
        sys.exit(f"ref detector '{ref_id}' unavailable: {why}")
    ad.load()
    return ad


def _crops_from_boxes(img: Image.Image, ref: Adapter, max_crops: int) -> list[Image.Image]:
    """Deskewed crops from an already-loaded ref detector's boxes on one page."""
    from app.geometry import deskew_crop
    from scanlation_sdk.contracts import Region
    crops: list[Image.Image] = []
    for b in ref.detect(img)[:max_crops]:
        if b.polygon and len(b.polygon) == 4:  # rotated quad -> deskew upright
            import numpy as np
            quad = np.array(b.polygon, dtype=np.float32)
            crops.append(deskew_crop(img, Region.from_quad(quad, angle=0.0, vertical=False)))
        else:
            crops.append(img.crop(tuple(int(v) for v in b.xyxy)))
    return crops


def _crops_from_ref(img: Image.Image, ref_id: str, max_crops: int) -> list[Image.Image]:
    """The SAME crops every OCR model reads, from the reference detector (default
    the decided box model) — so the comparison is apples-to-apples."""
    return _crops_from_boxes(img, _load_ref_detector(ref_id), max_crops)


def _ocr_devices(choice: str | None) -> list[str | None]:
    """Devices each OCR engine is run on. 'both' = CPU and (if present) CUDA in one
    run so the speed gap is side by side; explicit 'cpu'/'cuda' forces one; None =
    auto (best available), the old behaviour. Falls back to CPU with a note when
    CUDA is asked for but absent."""
    if choice == "both":
        devs: list[str | None] = ["cpu"]
        if _torch_device() == "cuda":
            devs.append("cuda")
        else:
            print("note: no CUDA device — 'both' shows CPU only", file=sys.stderr)
        return devs
    if choice == "cuda" and _torch_device() != "cuda":
        print("note: no CUDA device — running on CPU instead", file=sys.stderr)
        return ["cpu"]
    if choice in ("cpu", "cuda"):
        return [choice]
    return [None]  # auto: whatever the engine picks


def _run_label(adapter: Adapter, dev: str | None) -> str:
    """Column name for one (engine, device) run: the resolved device for engines we
    drive, or 'server' for ollama (its CPU/GPU choice is the server's business)."""
    return _resolve_device(dev) if adapter.device_switchable else "server"


def _time_recognize(adapter: Adapter, crops: list[Image.Image]) -> tuple[list[str], float]:
    """OCR every crop, returning (texts, total_ms). One warm-up call first so the
    timing excludes one-off init (CUDA kernels / lazy weights) and CPU-vs-GPU is fair."""
    if crops:
        adapter.recognize(crops[0])  # warm up, untimed
    t0 = time.perf_counter()
    texts = [adapter.recognize(c) for c in crops]
    return texts, (time.perf_counter() - t0) * 1000


def _write_ocr_report(out_path, crops: list[Image.Image], order: list[str],
                      texts: dict[str, list[str]], timings: dict[str, dict[str, float]],
                      *, quiet: bool = False) -> None:
    """Two markdown tables: a timing summary (engine x device: total + ms/crop, warm)
    then the transcribed text (one row per crop, one column per engine). quiet skips
    the stdout echo (used by the batch, which writes one report per image)."""
    engines = [e for e in order if e in texts]  # ran + produced text at least once
    dev_cols = [d for d in ("cpu", "cuda", "server")
                if any(d in timings.get(e, {}) for e in engines)]
    n = max(1, len(crops))

    L = [f"## OCR timing — {len(crops)} crops, warm (excludes first-call init)\n",
         "| engine | " + " | ".join(dev_cols) + " |",
         "|---|" + "|".join("---" for _ in dev_cols) + "|"]
    for e in engines:
        cells = []
        for d in dev_cols:
            ms = timings.get(e, {}).get(d)
            cells.append(f"{ms:.0f}ms ({ms / n:.1f}/crop)" if ms is not None else "—")
        L.append(f"| {e} | " + " | ".join(cells) + " |")
    if "cpu" in dev_cols and "cuda" in dev_cols:  # explicit speedup line per engine
        L.append("")
        for e in engines:
            c, g = timings.get(e, {}).get("cpu"), timings.get(e, {}).get("cuda")
            if c and g:
                L.append(f"- **{e}**: cuda {c / g:.1f}x faster than cpu ({c:.0f}ms -> {g:.0f}ms)")

    L += ["", "## OCR text — crops from ref detector\n",
          "| # | " + " | ".join(engines) + " |",
          "|---|" + "|".join("---" for _ in engines) + "|"]
    for i in range(len(crops)):
        cells = [(texts[e][i] if i < len(texts[e]) else "").replace("|", "\\|").replace("\n", " ") for e in engines]
        L.append(f"| {i:02d} | " + " | ".join(cells) + " |")

    report = "\n".join(L) + "\n"
    Path(out_path).write_text(report, encoding="utf-8")
    if not quiet:
        print("\n" + report)


def cmd_ocr(args) -> None:
    img = Image.open(args.image).convert("RGB")
    crops = _crops_from_ref(img, args.ref_detector, args.max_crops)
    print(f"\n{len(crops)} crops from ref detector '{args.ref_detector}'", file=sys.stderr)

    crop_dir = Path(args.crops)
    crop_dir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(crops):
        c.save(crop_dir / f"crop_{i:02d}.png")

    devices = _ocr_devices(args.device)
    print(f"OCR on device(s): {', '.join(str(d or 'auto') for d in devices)}"
          f"   (torch cuda: {'yes' if _torch_device() == 'cuda' else 'no'})", file=sys.stderr)

    order: list[str] = []                       # engine ids, run order (table column order)
    texts: dict[str, list[str]] = {}            # id -> per-crop text (device-independent)
    timings: dict[str, dict[str, float]] = {}   # id -> {device_label: total_ms}
    for a, ok, _ in _run_available(_select("ocr", args.only, args.exclude), _explicit_ids(args.only)):
        if not ok:
            continue
        run_devs = devices if a.device_switchable else [None]  # ollama: server picks the device
        run_devs = [dv for dv in run_devs if a.cpu_ok or dv != "cpu"]  # skip cpu for cpu_ok=False (e.g. 7B)
        if not run_devs:
            print(f"    {a.id}: skipped — needs a GPU but this run has no cuda device", file=sys.stderr)
            continue
        order.append(a.id)
        for dev in run_devs:
            a.device = dev
            lab = _run_label(a, dev)
            try:
                a.load()  # re-load per device (that IS the CPU-vs-GPU comparison)
                out, ms = _time_recognize(a, crops)
                texts.setdefault(a.id, out)  # text is device-independent; keep the first run's
                timings.setdefault(a.id, {})[lab] = ms
                print(f"    {a.id}[{lab}]: {ms:.0f}ms ({ms / max(1, len(crops)):.1f}ms/crop)", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 - one broken engine/device must not kill the run
                print(f"    {a.id}[{lab}]: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _write_ocr_report(args.out, crops, order, texts, timings)
    print(f"\nwrote {args.out}  +  {len(crops)} crops -> {crop_dir}/", file=sys.stderr)


def _write_ocr_summary(out_path: Path, n_images: int, total_crops: int,
                       order: list[str], agg: dict[str, dict[str, list]]) -> None:
    """Top-level CPU-vs-GPU speed table for the batch: total + ms/crop per engine
    and device, summed over every image (agg[engine][device] = [sum_ms, n_crops])."""
    engines = [e for e in order if e in agg]
    dev_cols = [d for d in ("cpu", "cuda", "server") if any(d in agg.get(e, {}) for e in engines)]
    L = [f"# OCR comparison — {n_images} images, {total_crops} crops (crops from the box model)\n",
         "Total OCR time per engine/device, warm (excludes first-call init).\n",
         "| engine | " + " | ".join(dev_cols) + " |",
         "|---|" + "|".join("---" for _ in dev_cols) + "|"]
    for e in engines:
        cells = []
        for d in dev_cols:
            v = agg.get(e, {}).get(d)
            cells.append(f"{v[0]:.0f}ms ({v[0] / max(1, v[1]):.1f}/crop)" if v else "—")
        L.append(f"| {e} | " + " | ".join(cells) + " |")
    if "cpu" in dev_cols and "cuda" in dev_cols:
        L.append("")
        for e in engines:
            c, g = agg.get(e, {}).get("cpu"), agg.get(e, {}).get("cuda")
            if c and g and g[0]:
                L.append(f"- **{e}**: cuda {c[0] / g[0]:.1f}x faster than cpu")
    report = "\n".join(L) + "\n"
    Path(out_path).write_text(report, encoding="utf-8")
    print("\n" + report)


def cmd_ocrbatch(args) -> None:
    """OCR every image under a folder tree, model-outer (each engine loads once per
    device and is reused across all images). Crops come from the ref detector (the
    decided box model). Output mirrors the input tree, taking the detection outputs'
    old place:  <out>/<category>/<image>/{ocr.md, ocr.json} (+ crops/), plus a
    top-level _ocr_summary.md with the CPU-vs-GPU speed table over all images.

    Results ACCUMULATE: each run merges the engines it produced into the per-image
    ocr.json, so a new engine can be added later (a fresh dep, a heavy GPU-only
    model, ollama) with `--only that_engine` without re-running — or losing — the
    others. (Prior results are reused only when the ref detector and crop count match.)"""
    import json
    root, out_root = Path(args.root), Path(args.out)
    tasks = _collect_tasks(root)
    if not tasks:
        sys.exit(f"no images found under {root}")
    devices = _ocr_devices(args.device)
    print(f"{len(tasks)} images under {root}; ref='{args.ref_detector}'; "
          f"OCR device(s): {', '.join(str(d or 'auto') for d in devices)}\n", file=sys.stderr)

    # 1) ref detector loads once; crops per image saved + cached. Any prior ocr.json
    #    (same ref + crop count) is loaded so its engine columns survive this run.
    ref = _load_ref_detector(args.ref_detector)
    crops_by_img: dict[tuple[str, str], list[Image.Image]] = {}
    prev: dict[tuple[str, str], dict] = {}
    for cat, img_path in tasks:
        key = (cat, img_path.stem)
        crops = _crops_from_boxes(Image.open(img_path).convert("RGB"), ref, args.max_crops)
        crops_by_img[key] = crops
        d = out_root / cat / img_path.stem
        (d / "crops").mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(crops):
            c.save(d / "crops" / f"crop_{i:02d}.png")
        jf = d / "ocr.json"
        if jf.exists():
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                if data.get("ref") == args.ref_detector and data.get("n_crops") == len(crops):
                    prev[key] = data.get("engines", {})
                else:
                    print(f"  note: {cat}/{img_path.stem}: prior ocr.json dropped (ref/crop-count changed)", file=sys.stderr)
            except Exception:  # noqa: BLE001
                pass
    total_crops = sum(len(v) for v in crops_by_img.values())
    print(f"ref '{args.ref_detector}': {total_crops} crops over {len(tasks)} images\n", file=sys.stderr)

    # 2) each engine x device, model-outer: load once, then read every image's crops
    this_run: dict[tuple[str, str], dict[str, dict]] = {}  # img -> engine -> {"texts":[...], "timings":{dev:ms}}
    for a, ok, _ in _run_available(_select("ocr", args.only, args.exclude), _explicit_ids(args.only)):
        if not ok:
            continue
        run_devs = devices if a.device_switchable else [None]  # ollama: server picks the device
        run_devs = [dv for dv in run_devs if a.cpu_ok or dv != "cpu"]  # skip cpu for cpu_ok=False (e.g. 7B)
        if not run_devs:
            print(f"  {a.id}: skipped — needs a GPU (cpu_ok=False) but this run has no cuda device", file=sys.stderr)
            continue
        for dev in run_devs:
            a.device = dev
            lab = _run_label(a, dev)
            try:
                a.load()
            except Exception as exc:  # noqa: BLE001
                print(f"  {a.id}[{lab}]: LOAD ERROR {type(exc).__name__}: {exc} — skipping", file=sys.stderr)
                continue
            warmed, run_row = False, [0.0, 0]
            for cat, img_path in tasks:
                key = (cat, img_path.stem)
                crops = crops_by_img[key]
                if not crops:
                    continue
                try:
                    if not warmed:
                        a.recognize(crops[0])  # one warm-up per (engine, device), untimed
                        warmed = True
                    t0 = time.perf_counter()
                    out = [a.recognize(c) for c in crops]
                    ms = (time.perf_counter() - t0) * 1000
                    e = this_run.setdefault(key, {}).setdefault(a.id, {"texts": None, "timings": {}})
                    if e["texts"] is None:
                        e["texts"] = out  # text is device-independent; keep the first device's
                    e["timings"][lab] = ms
                    run_row[0] += ms
                    run_row[1] += len(crops)
                except Exception as exc:  # noqa: BLE001 - one broken image must not kill the run
                    print(f"    {cat}/{img_path.stem} {a.id}[{lab}]: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            if run_row[1]:
                print(f"  {a.id}[{lab}]: {run_row[0]:.0f}ms over {run_row[1]} crops "
                      f"({run_row[0] / run_row[1]:.1f}ms/crop)", file=sys.stderr)

    # 3) merge prior + this run per image; rewrite ocr.md + ocr.json; feed the summary
    canon = [a.id for a in all_adapters() if a.kind == "ocr"]  # stable column order
    global_agg: dict[str, dict[str, list]] = {}
    for cat, img_path in tasks:
        key = (cat, img_path.stem)
        merged = {**prev.get(key, {}), **this_run.get(key, {})}  # this run's engines win
        if not merged:
            continue
        order = [e for e in canon if e in merged]
        texts = {e: merged[e]["texts"] for e in order}
        timings = {e: merged[e]["timings"] for e in order}
        d = out_root / cat / img_path.stem
        _write_ocr_report(d / "ocr.md", crops_by_img[key], order, texts, timings, quiet=True)
        (d / "ocr.json").write_text(json.dumps(
            {"ref": args.ref_detector, "n_crops": len(crops_by_img[key]), "engines": merged},
            ensure_ascii=False, indent=1), encoding="utf-8")
        for e in order:  # accumulate the summary over all images
            for dev, ms in timings[e].items():
                row = global_agg.setdefault(e, {}).setdefault(dev, [0.0, 0])
                row[0] += ms
                row[1] += len(crops_by_img[key])

    _write_ocr_summary(out_root / "_ocr_summary.md", len(tasks), total_crops,
                       [e for e in canon if e in global_agg], global_agg)
    print(f"\nwrote {out_root}/<category>/<image>/{{ocr.md, ocr.json}}  +  {out_root}/_ocr_summary.md", file=sys.stderr)


def _collect_tasks(root: Path) -> list[tuple[str, Path]]:
    """(category, image_path) for every image under root, category = its subfolder
    (so samples/노이즈/x.jpg -> ('노이즈', .../x.jpg)). Loose images -> '_root'."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    tasks = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            rel = p.relative_to(root).parent
            tasks.append((str(rel).replace("\\", "/") if str(rel) != "." else "_root", p))
    return tasks


def cmd_batch(args) -> None:
    """Run every available detector over a folder tree, model-outer so each model
    loads once and is reused across all images. Output mirrors the input tree:
    <out>/<category>/<image_stem>/<model>.png  (+ a _montage.png per image)."""
    root, out_root = Path(args.root), Path(args.out)
    tasks = _collect_tasks(root)
    if not tasks:
        sys.exit(f"no images found under {root}")
    print(f"{len(tasks)} images under {root} -> {out_root}/<category>/<image>/<model>.png\n", file=sys.stderr)

    ads = _select("detect", args.only, args.exclude)
    _apply_opts(ads, args.opt)
    adapters = [a for a, ok, _ in _run_available(ads, _explicit_ids(args.only)) if ok]
    if not adapters:
        sys.exit("no detectors available (run `list`)")

    counts: dict[tuple[str, str], dict[str, int]] = {}
    for a in adapters:
        try:
            a.load()
        except Exception as exc:  # noqa: BLE001
            print(f"  {a.id}: LOAD ERROR {type(exc).__name__}: {exc} — skipping", file=sys.stderr)
            continue
        print(f"  {a.id}: running on {len(tasks)} images ...", file=sys.stderr)
        for cat, img_path in tasks:
            try:
                img = Image.open(img_path).convert("RGB")
                t0 = time.perf_counter()
                boxes = a.detect(img)
                ms = (time.perf_counter() - t0) * 1000
                panel = render_panel(img, DetResult(boxes, ms), a.label, by_class=args.by_class)
                d = out_root / cat / img_path.stem
                d.mkdir(parents=True, exist_ok=True)
                panel.save(d / f"{a.id}.png")
                counts.setdefault((cat, img_path.stem), {})[a.id] = len(boxes)
            except Exception as exc:  # noqa: BLE001
                print(f"    {cat}/{img_path.name} {a.id}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)

    for cat, img_path in tasks:  # a side-by-side montage per image, from the saved panels
        d = out_root / cat / img_path.stem
        panels = [Image.open(d / f"{a.id}.png") for a in adapters if (d / f"{a.id}.png").exists()]
        if panels:
            montage(panels, d / "_montage.png")

    ids = [a.id for a in adapters]
    print("\n=== box counts per image (category / image) ===")
    last = None
    for (cat, stem), c in sorted(counts.items()):
        if cat != last:
            print(f"\n[{cat}]")
            last = cat
        print(f"  {stem}: " + "  ".join(f"{i}={c.get(i, '-')}" for i in ids))
    print(f"\nwrote {out_root}/  ({len(tasks)} images x {len(adapters)} models)", file=sys.stderr)


def cmd_ba(args) -> None:
    """Before/after for ONE detector's dedup: per image render it with dedup OFF
    (before.png) and ON (after.png) into <out>/<category>/<image>/, keeping the
    input tree. 'After' reuses 'before's boxes deduped (one inference per image);
    the after thresholds are the adapter's configured nms_iou/contain_thresh,
    overridable with --opt (e.g. --opt nms_iou=0.5)."""
    root, out_root = Path(args.root), Path(args.out)
    tasks = _collect_tasks(root)
    if not tasks:
        sys.exit(f"no images found under {root}")
    ads = _select("detect", args.detector, None)
    _apply_opts(ads, args.opt)
    ad = next((a for a in ads if a.id == args.detector), None)
    if ad is None:
        sys.exit(f"unknown detector: {args.detector} (run `list`)")
    ok, why = ad.available()
    if not ok:
        sys.exit(f"{args.detector} unavailable: {why}")
    ad.load()
    after_iou, after_contain = getattr(ad, "nms_iou", None), getattr(ad, "contain_thresh", None)
    if after_iou is None and after_contain is None:
        print(f"note: {args.detector} has no dedup configured — after == before", file=sys.stderr)
    conf = getattr(ad, "conf", "?")
    before_title = f"BEFORE  {args.detector}  conf={conf}  dedup=off"
    after_title = f"AFTER  {args.detector}  conf={conf}  nms_iou={after_iou}  contain={after_contain}"
    print(f"{len(tasks)} images: {args.detector}  before(off) vs after(nms_iou={after_iou}, contain_thresh={after_contain})\n",
          file=sys.stderr)
    for cat, img_path in tasks:
        img = Image.open(img_path).convert("RGB")
        if hasattr(ad, "nms_iou"):  # force raw for 'before'
            ad.nms_iou, ad.contain_thresh = None, None
        before = ad.detect(img)
        after = dedup_boxes(before, after_iou, after_contain)
        d = out_root / cat / img_path.stem
        d.mkdir(parents=True, exist_ok=True)
        render_panel(img, DetResult(before, 0.0), before_title, by_class=args.by_class).save(d / "before.png")
        render_panel(img, DetResult(after, 0.0), after_title, by_class=args.by_class).save(d / "after.png")
        print(f"  {cat}/{img_path.stem}: {len(before)} -> {len(after)}")
    print(f"\nwrote {out_root}/<category>/<image>/{{before,after}}.png  ({len(tasks)} images)", file=sys.stderr)


def _consolidate_images(out_root: Path):
    """(rel, engine-cols, rows) per image from per-image ocr.json under out_root;
    rows[i] = each engine's text for crop i (aligned to the shared rtdetr crops)."""
    import json
    canon = [a.id for a in all_adapters() if a.kind == "ocr"]  # stable column order
    images = []
    for jf in sorted(out_root.rglob("ocr.json")):
        eng = json.loads(jf.read_text(encoding="utf-8")).get("engines", {})
        cols = [e for e in canon if e in eng]
        if not cols:
            continue
        n = max(len(eng[e]["texts"]) for e in cols)
        rows = [[(eng[e]["texts"][i] if i < len(eng[e]["texts"]) else "") for e in cols] for i in range(n)]
        images.append((jf.parent.relative_to(out_root).as_posix(), cols, rows))
    return images


def _write_ocr_md(dest: Path, images) -> None:
    esc = lambda s: (s or "").replace("|", "\\|").replace("\n", " ")  # noqa: E731
    L = [f"# OCR text comparison — {len(images)} images (crop recognizers, rows = rtdetr crops)\n"]
    for rel, cols, rows in images:
        L.append(f"\n## {rel}\n")
        L.append("| # | " + " | ".join(cols) + " |")
        L.append("|---|" + "|".join("---" for _ in cols) + "|")
        for i, row in enumerate(rows):
            L.append(f"| {i:02d} | " + " | ".join(esc(t) for t in row) + " |")
    dest.write_text("\n".join(L) + "\n", encoding="utf-8")


def _diff_spans(ref: str, s: str) -> str:
    """s shown verbatim (whitespace kept), but only the NON-whitespace runs that differ
    from ref are wrapped .d (red). Whitespace is ignored when diffing, so VLM-inserted
    spaces don't count as differences."""
    import difflib
    import html

    def strip(t):  # non-whitespace chars + each one's original index in t
        idx = [i for i, c in enumerate(t) if not c.isspace()]
        return "".join(t[i] for i in idx), idx

    rs, _ = strip(ref)
    ss, spos = strip(s)
    diff = [False] * len(s)  # per original char of s; whitespace stays False (plain)
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(None, rs, ss, autojunk=False).get_opcodes():
        if tag != "equal":
            for j in range(j1, j2):
                diff[spos[j]] = True
    out, k = [], 0
    while k < len(s):  # group consecutive same-flag original chars into spans
        m = k
        while m < len(s) and diff[m] == diff[k]:
            m += 1
        seg = html.escape(s[k:m])
        out.append(f'<span class="d">{seg}</span>' if diff[k] else seg)
        k = m
    return "".join(out)


_HTML_JS = """
(function(){
  var K=VK, tally={};  // VK injected per page ('ocrsel:' for OCR, 'boxsel:' for BOX) -> separate vote namespaces
  function refresh(){
    var t=document.getElementById('tally');
    t.innerHTML='선택수 — '+engs.map(function(e){return '<b>'+e+'</b> '+(tally[e]||0);}).join(' &nbsp;·&nbsp; ');
    var cb=document.getElementById('catbreak'); if(!cb) return;
    var cat={};  // category -> engine -> votes, aggregated live from the selected cells
    document.querySelectorAll('.eng.sel').forEach(function(td){
      var c=td.getAttribute('data-cat'), e=td.getAttribute('data-eng');
      (cat[c]=cat[c]||{})[e]=(cat[c][e]||0)+1;
    });
    var h='<table class="cb"><tr><th>분류</th><th>crops</th>'+engs.map(function(e){return '<th>'+e+'</th>';}).join('')+'</tr>';
    var tot={}, totN=0;
    catList.forEach(function(c){
      var row=cat[c]||{}, n=catN[c]||0; totN+=n;
      var best=0; engs.forEach(function(e){ if((row[e]||0)>best) best=row[e]||0; });
      h+='<tr><td class="cn">'+c+'</td><td class="cc">'+n+'</td>'+engs.map(function(e){
        var v=row[e]||0; tot[e]=(tot[e]||0)+v;
        return '<td'+((v===best&&v>0)?' class="win"':'')+'>'+v+'<i>'+(n?Math.round(100*v/n):0)+'%</i></td>';
      }).join('')+'</tr>';
    });
    var tb=0; engs.forEach(function(e){ if((tot[e]||0)>tb) tb=tot[e]||0; });  // overall winner
    h+='<tr class="tot"><td class="cn">합계</td><td class="cc">'+totN+'</td>'+engs.map(function(e){
      var v=tot[e]||0;
      return '<td'+((v===tb&&v>0)?' class="win"':'')+'>'+v+'<i>'+(totN?Math.round(100*v/totN):0)+'%</i></td>';
    }).join('')+'</tr></table>';
    cb.innerHTML=h;
  }
  function set(td,on){
    var e=td.getAttribute('data-eng'), k=K+td.getAttribute('data-key');
    if(on){td.classList.add('sel');tally[e]=(tally[e]||0)+1;try{localStorage.setItem(k,'1')}catch(x){}}
    else {td.classList.remove('sel');tally[e]=(tally[e]||0)-1;try{localStorage.removeItem(k)}catch(x){}}
  }
  document.addEventListener('click',function(ev){
    var td=ev.target.closest&&ev.target.closest('.eng'); if(!td) return;
    set(td,!td.classList.contains('sel')); refresh();
  });
  document.querySelectorAll('.eng').forEach(function(td){
    var on=false; try{on=localStorage.getItem(K+td.getAttribute('data-key'))==='1'}catch(x){}
    if(on){td.classList.add('sel');var e=td.getAttribute('data-eng');tally[e]=(tally[e]||0)+1;}
  });
  var rb=document.getElementById('reset');
  if(rb) rb.addEventListener('click',function(){
    document.querySelectorAll('.eng.sel').forEach(function(td){set(td,false)}); refresh();
  });
  refresh();
})();
"""


def _write_ocr_html(dest: Path, images, ref_id: str, out_root: Path, *, embed: bool = True, cap: int = 400) -> None:
    """Self-contained HTML: per image a table of crop rows — the crop image, then each
    engine's text with only the runs that DIFFER from ref_id highlighted red. Engine
    cells are clickable to tally a manual "who read this crop best" vote per model (live
    count in the sticky bar, persisted in localStorage). Crop images are base64-embedded
    (portable) unless embed=False (relative <img src>)."""
    import base64
    import html
    import json
    css = ("body{font-family:'Segoe UI',system-ui,sans-serif;margin:14px;background:#1e1e1e;color:#d4d4d4}"
           "h2{margin:22px 0 4px;font-size:14px;color:#e0e0e0;border-bottom:1px solid #3a3a3a}"
           ".legend{position:sticky;top:0;background:#1e1e1e;padding:8px 0;border-bottom:1px solid #444;font-size:13px;z-index:3}"
           "#tally{margin-left:6px;color:#cfe8ff}#reset{margin-left:10px;font-size:12px;cursor:pointer;background:#333;color:#ddd;border:1px solid #555;border-radius:3px;padding:1px 8px}"
           ".d{background:#6e1f24;color:#ffd0d0;font-weight:600}"
           "table{border-collapse:collapse;width:100%;table-layout:fixed;margin-bottom:6px}"
           "th,td{border:1px solid #3a3a3a;padding:4px 6px;vertical-align:top;font-size:13px;word-break:break-word;line-height:1.55}"
           "th{background:#2d2d2d;color:#e0e0e0;position:sticky;top:36px}"
           "td.idx,th.idx{width:30px;text-align:center;color:#888}td.ref{background:#262626}"
           "td.eng{cursor:pointer}td.eng:hover{outline:1px solid #4a5a6a}td.sel{box-shadow:inset 0 0 0 2px #4fc3f7;background:#1e3a5f !important}"
           "td.im,th.im{width:210px}img{max-width:200px;max-height:170px;object-fit:contain;display:block;background:#f5f5f5;border:1px solid #555}"
           "details#catwrap{margin:6px 0 2px}summary{cursor:pointer;color:#cfe8ff;font-size:13px}"
           "table.cb{width:auto;border-collapse:collapse;margin:6px 0 10px;font-size:12px;table-layout:auto}"
           "table.cb th,table.cb td{border:1px solid #3a3a3a;padding:3px 9px;text-align:center;position:static}"
           "table.cb td.cn{text-align:left;color:#e0e0e0}table.cb td.cc{color:#888}"
           "table.cb td.win{background:#4a3b12;color:#ffe9a8;font-weight:700}"
           "table.cb i{color:#7f93a6;font-style:normal;font-size:10px;margin-left:4px}"
           "table.cb tr.tot td{border-top:2px solid #5a5a5a;font-weight:600}")
    clip = lambda t: t if len(t) <= cap else t[:cap] + f" …(+{len(t) - cap})"  # noqa: E731
    esc_attr = lambda t: html.escape(t, quote=True)  # noqa: E731

    def crop_cell(rel: str, i: int) -> str:
        p = out_root / rel / "crops" / f"crop_{i:02d}.png"
        if not p.exists():
            return "<span style='color:#777'>—</span>"
        src = ("data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()) if embed \
            else f"{rel}/crops/crop_{i:02d}.png"
        return f"<img src='{src}' loading='lazy'>"

    engs = []  # union of engines in encounter order, for the tally
    for _rel, cols, _rows in images:
        for c in cols:
            if c not in engs:
                engs.append(c)
    cat_n, cat_list = {}, []  # crops per category (denominator for per-category acceptance %)
    for rel, _cols, rows in images:
        c = rel.split("/")[0]
        if c not in cat_list:
            cat_list.append(c)
        cat_n[c] = cat_n.get(c, 0) + len(rows)
    P = [f"<!doctype html><html lang='ja'><head><meta charset='utf-8'><title>OCR compare</title><style>{css}</style></head><body>",
         f"<div class='legend'>기준 = <b>{html.escape(ref_id)}</b> · 차이만 <span class='d'>&nbsp;빨강&nbsp;</span> "
         f"(공백 무시) · 칸 클릭 = 득표 <span id='tally'></span><button id='reset'>초기화</button></div>",
         "<details open id='catwrap'><summary>분류별 채택률 (분류 × 엔진 — 득표수 · %)</summary>"
         "<div id='catbreak'></div></details>"]
    for rel, cols, rows in images:
        ri = cols.index(ref_id) if ref_id in cols else 0
        cat = rel.split("/")[0]  # category = first path segment, embedded in each cell for per-category tally
        P.append(f"<h2>{html.escape(rel)}</h2><table><tr><th class='idx'>#</th><th class='im'>crop</th>"
                 + "".join(f"<th>{html.escape(c)}{' (기준)' if i == ri else ''}</th>" for i, c in enumerate(cols)) + "</tr>")
        for i, row in enumerate(rows):
            ref = clip(row[ri])
            tds = []
            for j, txt in enumerate(row):
                inner = html.escape(clip(txt)) if j == ri else _diff_spans(ref, clip(txt))
                cls = "eng ref" if j == ri else "eng"
                tds.append(f"<td class='{cls}' data-eng='{esc_attr(cols[j])}' data-cat='{esc_attr(cat)}' "
                           f"data-key='{esc_attr(f'{rel}|{i:02d}|{cols[j]}')}'>{inner}</td>")
            P.append(f"<tr><td class='idx'>{i:02d}</td><td class='im'>{crop_cell(rel, i)}</td>" + "".join(tds) + "</tr>")
        P.append("</table>")
    P.append(f"<script>var VK='ocrsel:',engs={json.dumps(engs)},"
             f"catList={json.dumps(cat_list, ensure_ascii=False)},"
             f"catN={json.dumps(cat_n, ensure_ascii=False)};{_HTML_JS}</script></body></html>")
    dest.write_text("".join(P), encoding="utf-8")


def _consolidate_box_images(out_root: Path):
    """(rel, [model ids present]) per image, from the <model>.png box-overlays that
    `batch` writes under out_root/<cat>/<img>/. Models kept in canonical detector order."""
    det_ids = [a.id for a in all_adapters() if a.kind == "detect"]
    dirs = sorted({p.parent for m in det_ids for p in out_root.rglob(f"{m}.png")})
    images = []
    for d in dirs:
        present = [m for m in det_ids if (d / f"{m}.png").exists()]
        if present:
            images.append((d.relative_to(out_root).as_posix(), present))
    return images


def _write_box_html(dest: Path, images, out_root: Path, *, embed: bool = False) -> None:
    """Self-contained HTML to score DETECTORS (sibling of _write_ocr_html): per image,
    each model's box-overlay side by side, each panel clickable to vote 'this model boxed
    the page best'. Per-model tally + per-category matrix, persisted in localStorage under
    the boxsel: namespace (separate from OCR votes). Overlays linked by relative path by
    default (full-page PNGs are big); embed=True base64-inlines them (portable, heavy)."""
    import base64
    import html
    import json
    css = ("body{font-family:'Segoe UI',system-ui,sans-serif;margin:14px;background:#1e1e1e;color:#d4d4d4}"
           "h2{margin:22px 0 6px;font-size:15px;color:#e0e0e0;border-bottom:1px solid #3a3a3a}"
           "h3{margin:14px 0 4px;font-size:13px;color:#bbb;font-weight:600}"
           ".legend{position:sticky;top:0;background:#1e1e1e;padding:8px 0;border-bottom:1px solid #444;font-size:13px;z-index:3}"
           "#tally{margin-left:6px;color:#cfe8ff}#reset{margin-left:10px;font-size:12px;cursor:pointer;background:#333;color:#ddd;border:1px solid #555;border-radius:3px;padding:1px 8px}"
           "details#catwrap{margin:6px 0 2px}summary{cursor:pointer;color:#cfe8ff;font-size:13px}"
           "table.cb{width:auto;border-collapse:collapse;margin:6px 0 10px;font-size:12px}"
           "table.cb th,table.cb td{border:1px solid #3a3a3a;padding:3px 9px;text-align:center}"
           "table.cb td.cn{text-align:left;color:#e0e0e0}table.cb td.cc{color:#888}"
           "table.cb td.win{background:#4a3b12;color:#ffe9a8;font-weight:700}"
           "table.cb i{color:#7f93a6;font-style:normal;font-size:10px;margin-left:4px}"
           "table.cb tr.tot td{border-top:2px solid #5a5a5a;font-weight:600}"
           ".row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px}"
           ".eng.box{width:360px;border:1px solid #3a3a3a;border-radius:4px;padding:3px;cursor:pointer;background:#232323}"
           ".eng.box:hover{outline:1px solid #4a5a6a}.eng.box.sel{box-shadow:inset 0 0 0 3px #4fc3f7;background:#1e3a5f}"
           ".ml{font-size:12px;color:#cfe8ff;padding:2px 4px}"
           ".eng.box img{width:100%;display:block;background:#f5f5f5;border-radius:2px}")
    esc_a = lambda t: html.escape(t, quote=True)  # noqa: E731

    engs = []
    for _rel, models in images:
        for m in models:
            if m not in engs:
                engs.append(m)
    cat_n, cat_list = {}, []  # denominator = images per category (one overlay set per image)
    for rel, _m in images:
        c = rel.split("/")[0]
        if c not in cat_list:
            cat_list.append(c)
        cat_n[c] = cat_n.get(c, 0) + 1

    def src(rel: str, m: str):
        p = out_root / rel / f"{m}.png"
        if not p.exists():
            return None
        return ("data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()) if embed else f"{rel}/{m}.png"

    P = [f"<!doctype html><html lang='ja'><head><meta charset='utf-8'><title>BOX compare</title><style>{css}</style></head><body>",
         "<div class='legend'>detector 박스 채점 · 오버레이 클릭 = 득표 "
         "<span id='tally'></span><button id='reset'>초기화</button></div>",
         "<details open id='catwrap'><summary>분류별 채택률 (분류 × 모델 — 득표수 · %)</summary>"
         "<div id='catbreak'></div></details>"]
    last_cat = None
    for rel, _models in images:
        cat = rel.split("/")[0]
        if cat != last_cat:
            P.append(f"<h2>{html.escape(cat)}</h2>")
            last_cat = cat
        P.append(f"<h3>{html.escape(rel.split('/', 1)[-1])}</h3><div class='row'>")
        for m in engs:
            s = src(rel, m)
            if s is None:
                continue
            P.append(f"<div class='eng box' data-eng='{esc_a(m)}' data-cat='{esc_a(cat)}' "
                     f"data-key='{esc_a(f'{rel}|{m}')}'><div class='ml'>{html.escape(m)}</div>"
                     f"<img src='{s}' loading='lazy'></div>")
        P.append("</div>")
    P.append(f"<script>var VK='boxsel:',engs={json.dumps(engs)},"
             f"catList={json.dumps(cat_list, ensure_ascii=False)},"
             f"catN={json.dumps(cat_n, ensure_ascii=False)};{_HTML_JS}</script></body></html>")
    dest.write_text("".join(P), encoding="utf-8")


def cmd_consolidate(args) -> None:
    """Gather every image's crop-OCR into ONE comparison doc (md and/or html). Rows =
    the shared rtdetr crops, columns = engines, from per-image ocr.json under <out>.
    The html char-diffs each engine against --ref (shared text green, differences red)
    so agreement vs divergence is visible at a glance. (Page-level engines aren't in
    ocr.json, so only crop recognizers appear.)"""
    out_root = Path(args.out)
    images = _consolidate_images(out_root)
    if not images:
        sys.exit(f"no usable ocr.json under {out_root} (run ocrbatch first)")
    made = []
    if args.format in ("md", "both"):
        _write_ocr_md(out_root / f"{args.name}.md", images)
        made.append(f"{args.name}.md")
    if args.format in ("html", "both"):
        _write_ocr_html(out_root / f"{args.name}.html", images, args.ref, out_root, embed=not args.link)
        made.append(f"{args.name}.html")
    print(f"wrote {out_root}/{{{', '.join(made)}}}  ({len(images)} images)", file=sys.stderr)


def cmd_boxhtml(args) -> None:
    """Gather detector box-overlays (compare_out/<cat>/<img>/<model>.png from `batch`) into
    ONE scoring HTML — per image each model's overlay side by side, click a panel to vote,
    per-model + per-category tallies (localStorage, boxsel: namespace). BOX analog of consolidate."""
    out_root = Path(args.out)
    images = _consolidate_box_images(out_root)
    if not images:
        sys.exit(f"no detector overlays under {out_root} (run `batch` first)")
    _write_box_html(out_root / f"{args.name}.html", images, out_root, embed=args.embed)
    print(f"wrote {out_root}/{args.name}.html  ({len(images)} images)", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show every adapter + whether it can run here").set_defaults(fn=cmd_list)

    d = sub.add_parser("detect", help="run every available detector -> side-by-side montage")
    d.add_argument("image")
    d.add_argument("--out", default="compare_out/compare_detectors.png")
    d.add_argument("--only", default=None, help="comma ids, e.g. ogkalu_rtdetr,ogkalu_yolov8m")
    d.add_argument("--exclude", default=None, help="comma ids to drop, e.g. kitsumed_seg")
    d.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                   help="tune an adapter attr, e.g. --opt conf=0.5 --opt nms_iou=0.5 (repeatable)")
    d.add_argument("--panels", default="compare_out/compare_panels", help="dir for full-res per-detector images")
    d.add_argument("--color-by-class", dest="by_class", action="store_true",
                   help="color boxes per class (default: all red, like the sample)")
    d.set_defaults(fn=cmd_detect)

    o = sub.add_parser("ocr", help="run every available OCR on the ref detector's crops")
    o.add_argument("image")
    o.add_argument("--ref-detector", default="ogkalu_rtdetr",
                   help="detector that makes the crops (the decided box model)")
    o.add_argument("--device", default=None, choices=["cpu", "cuda", "both"],
                   help="run OCR on cpu, cuda, or both (side-by-side timing); default: auto")
    o.add_argument("--only", default=None, help="comma ids, e.g. mangaocr,qwen3vl")
    o.add_argument("--exclude", default=None, help="comma ids to drop")
    o.add_argument("--max-crops", type=int, default=20)
    o.add_argument("--out", default="compare_out/compare_ocr.md")
    o.add_argument("--crops", default="compare_out/crops_ocr")
    o.set_defaults(fn=cmd_ocr)

    ob = sub.add_parser("ocrbatch", help="run OCR over a folder tree -> <out>/<category>/<image>/ocr.md")
    ob.add_argument("root", nargs="?", default="samples", help="input root (default: samples)")
    ob.add_argument("--out", default="compare_out", help="output root (mirrors the input tree)")
    ob.add_argument("--ref-detector", default="ogkalu_rtdetr", help="detector that makes the crops (box model)")
    ob.add_argument("--device", default=None, choices=["cpu", "cuda", "both"],
                    help="run OCR on cpu, cuda, or both (side-by-side timing); default: auto")
    ob.add_argument("--only", default=None, help="comma ids, e.g. mangaocr,paddleocr_vl")
    ob.add_argument("--exclude", default=None, help="comma ids to drop")
    ob.add_argument("--max-crops", type=int, default=20)
    ob.set_defaults(fn=cmd_ocrbatch)

    cs = sub.add_parser("consolidate", help="gather per-image ocr.json into one crop-OCR comparison doc (md/html)")
    cs.add_argument("--out", default="compare_out", help="tree with per-image ocr.json (default: compare_out)")
    cs.add_argument("--name", default="_compare_crops", help="output file stem under --out")
    cs.add_argument("--format", default="both", choices=["md", "html", "both"], help="default: both")
    cs.add_argument("--ref", default="mangaocr", help="engine others are diff-highlighted against in html")
    cs.add_argument("--link", action="store_true",
                    help="html: reference crop images by relative path instead of base64-embedding (smaller file)")
    cs.set_defaults(fn=cmd_consolidate)

    bh = sub.add_parser("boxhtml", help="gather detector <model>.png overlays into one BOX-scoring HTML (click-to-vote)")
    bh.add_argument("--out", default="compare_out", help="tree with per-image <model>.png overlays (default: compare_out)")
    bh.add_argument("--name", default="_compare_box", help="output file stem under --out")
    bh.add_argument("--embed", action="store_true",
                    help="base64-embed overlays (default: link by relative path — full-page PNGs are large)")
    bh.set_defaults(fn=cmd_boxhtml)

    b = sub.add_parser("batch", help="run detectors over a folder tree -> <out>/<category>/<image>/<model>.png")
    b.add_argument("root", nargs="?", default="samples", help="input root of category folders (default: samples)")
    b.add_argument("--out", default="compare_out", help="output root (mirrors the input tree)")
    b.add_argument("--only", default=None, help="comma ids, e.g. ogkalu_rtdetr,kitsumed_seg")
    b.add_argument("--exclude", default=None, help="comma ids to drop, e.g. kitsumed_seg")
    b.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                   help="tune an adapter attr, e.g. --opt conf=0.5 --opt nms_iou=0.5 (repeatable)")
    b.add_argument("--color-by-class", dest="by_class", action="store_true",
                   help="color boxes per class (default: all red)")
    b.set_defaults(fn=cmd_batch)

    ba = sub.add_parser("ba", help="before/after dedup for one detector -> <out>/<cat>/<img>/{before,after}.png")
    ba.add_argument("root", nargs="?", default="samples", help="input root (default: samples)")
    ba.add_argument("--out", default="compare_out", help="output root (mirrors the input tree)")
    ba.add_argument("--detector", default="ogkalu_rtdetr", help="detector to A/B on dedup")
    ba.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                    help="override the AFTER dedup, e.g. --opt nms_iou=0.5 --opt contain_thresh=0.7")
    ba.add_argument("--color-by-class", dest="by_class", action="store_true",
                    help="color boxes per class (default: all red)")
    ba.set_defaults(fn=cmd_ba)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
