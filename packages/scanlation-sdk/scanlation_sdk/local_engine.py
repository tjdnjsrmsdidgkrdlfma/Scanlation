"""LocalModelEngineBase — shared lifecycle for engines whose weights live
locally and load into torch (comic-text-and-bubble-detector/manga-ocr/PaddleOCR-VL-For-Manga).

Centralizes what those engines genuinely duplicate: the load/install guards,
the "install() downloads, load() never does" project rule, device selection,
and CUDA cache release on unload. ``is_installed()`` stays per-engine — the
checks (local files vs HF cache probe) genuinely differ.

Plugin-facing only: the server core must not import this module (see device.py).
"""
from __future__ import annotations

from PIL import Image

from scanlation_sdk.contracts import EngineBase
from scanlation_sdk.device import device_label, pick_device, release_cuda_cache


def to_rgb(img: Image.Image) -> Image.Image:
    """An RGB view of ``img`` — ``img`` itself if already RGB, else a converted
    copy. Local recognizers/detectors feed models that expect 3 channels."""
    return img if img.mode == "RGB" else img.convert("RGB")


GRID = 28  # dynamic-res VLM vision patch grid (PaddleOCR-VL: patch14 x merge2)


def downscale_to_cap(crop: Image.Image, cap: int, mode: str = "pow2") -> Image.Image:
    """Shrink a crop to <= ``cap`` pixels (aspect preserved) so a dynamic-resolution
    VLM recognizer emits fewer vision tokens. ``pow2`` (BOX integer halving) is the
    validated-best mode — packages/scanlation-server/tools/recognize-gpu-speed.md.
    ``cap <= 0`` or an already-small crop is returned unchanged (same object)."""
    w, h = crop.width, crop.height
    if cap <= 0 or w * h <= cap:
        return crop
    if mode == "pow2":
        while crop.width * crop.height > cap and crop.width >= 2 and crop.height >= 2:
            crop = crop.reduce(2)
        return crop
    scale = (cap / (w * h)) ** 0.5
    tw, th = max(1, int(w * scale)), max(1, int(h * scale))
    if mode in ("grid28", "boxgrid"):
        tw, th = max(GRID, tw - tw % GRID), max(GRID, th - th % GRID)
    return crop.resize((tw, th), Image.BOX if mode in ("box", "boxgrid") else Image.LANCZOS)


def install_hint(name: str, extra: str = "") -> str:
    """The '<engine> weights not installed' tail: the two install routes for
    ``name``. Ends in '.'; pass ``extra`` — a clause carrying its own separator
    and terminator, e.g. a model-path env override — to replace that period."""
    return (
        f'Install first: POST /install_plugins/ {{"{name}": true}}, or '
        f"`python tools/install.py {name}`{extra or '.'}"
    )


class LocalModelEngineBase(EngineBase):
    # Per-engine "how to install" tail of the not-installed error.
    INSTALL_HINT: str = ""
    # Class default so a subclass that skips super().__init__() is still safe
    # (engine_meta.safe_is_installed instantiates throwaway cls()).
    _loaded: bool = False
    # Compute device this engine loads onto when the user sets no override —
    # the code default, like an OPTION_SCHEMA option's `default`. Subclasses
    # override (PaddleOCR-VL-For-Manga -> "cuda"); cpu-viable engines keep "cpu".
    DEFAULT_DEVICE: str = "cpu"
    # Per-engine device override injected by the registry from admin state;
    # None -> DEFAULT_DEVICE. Class default keeps super().__init__()-skipping
    # subclasses safe, same as _loaded.
    _device_override: str | None = None

    # --- subclass hooks ---
    def _download(self) -> None:
        """install()'s body: fetch weights (snapshot_download etc.)."""
        raise NotImplementedError

    def _load(self, device: str) -> None:
        """Acquire model/processor attributes on ``device``."""
        raise NotImplementedError

    def _unload(self) -> None:
        """Drop whatever _load() set."""
        raise NotImplementedError

    # --- shared lifecycle ---
    def install(self) -> None:
        if self.is_installed():
            return
        self._download()

    def load(self) -> None:
        if self._loaded:
            return
        if not self.is_installed():
            raise RuntimeError(f"{self.name} weights not installed. {self.INSTALL_HINT}")
        device = pick_device(self._device_override or self.DEFAULT_DEVICE)
        self._load(device)
        self._loaded = True
        # Uniform load line for every local engine (per-engine logger namespace kept).
        self._log.info("%s loaded on %s", self.display_name, device_label(device))

    def unload(self) -> None:
        self._unload()
        self._loaded = False
        release_cuda_cache()
