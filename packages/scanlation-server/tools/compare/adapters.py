"""Detection and Japanese-OCR model adapters for the compare_models harness.
Each lazy-imports its framework and reports (skip, reason) when deps/weights are
absent, so candidates can be enabled one at a time."""
from __future__ import annotations

import base64
import io
import os

from PIL import Image

from compare.core import (
    OCR_PROMPT, TOOLS_DIR, Adapter, Box, _hf_weight_file, _imports_ok,
    _resolve_device, _torch_device, dedup_boxes,
)


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
    """A transformers object-detection model (e.g. RT-DETRv2) via the Auto* API,
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
    install_hint = "pip install manga-ocr   (or: -e packages/scanlation-manga-ocr)"

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
        self.install_hint = ("vendored — download weights into tools/vendored/_mit_weights/ "
                             "(from zyddnys/manga-image-translator release beta-0.3; gitignored, not bundled)")

    def _ckpt(self) -> str:
        return "ocr-ctc.ckpt" if self.variant == "ctc" else "ocr_ar_48px.ckpt"

    def available(self) -> tuple[bool, str]:
        ok, why = _imports_ok("torch", "cv2")
        if not ok:
            return False, why
        wpath = TOOLS_DIR / "vendored" / "_mit_weights" / self._ckpt()
        return (True, "") if wpath.exists() else (False, f"missing weights {self._ckpt()} in tools/vendored/_mit_weights/")

    def load(self) -> None:
        from vendored._mit_ocr import MitOCR
        self._m = MitOCR(self.variant)
        self._m.load(_resolve_device(self.device))

    def recognize(self, crop: Image.Image) -> str:
        import numpy as np
        return self._m.recognize(np.array(crop.convert("RGB")))
