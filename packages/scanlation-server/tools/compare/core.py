"""Normalized result types, the adapter base, and geometry helpers for the
compare_models research harness."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# The tools/ dir (this file sits at tools/compare/core.py). Vendored MIT-OCR
# weights live under TOOLS_DIR/vendored/_mit_weights/.
TOOLS_DIR = Path(__file__).resolve().parent.parent


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
