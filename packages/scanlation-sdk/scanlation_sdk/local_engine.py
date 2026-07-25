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


# The downscale modes downscale_to_cap understands — the authoritative set a
# recognizer's OPTION_SCHEMA should defer to instead of hardcoding its own whitelist
# (an unknown mode falls back to pow2 below). Both average over the source area, which
# is what a bake-off found to matter (packages/scanlation-server/tools/recognize-gpu-speed.md):
# LANCZOS rings on thin strokes and lost to every BOX variant, so no LANCZOS mode is
# offered. They differ only in the scale factors allowed.
DOWNSCALE_MODES = ("pow2", "box")


def downscale_to_cap(crop: Image.Image, cap: int, mode: str = "pow2") -> Image.Image:
    """Shrink a crop to <= ``cap`` pixels (aspect preserved) so a dynamic-resolution
    VLM recognizer emits fewer vision tokens. ``cap <= 0`` or an already-small crop is
    returned unchanged (same object); an unrecognized ``mode`` falls back to ``pow2``.

    ``pow2`` halves repeatedly, so each output pixel is the exact mean of a 2^k block —
    but the only reachable areas are 1, 1/4, 1/16…, and a crop barely over the cap
    drops a whole step to a QUARTER of the budget. ``box`` computes the scale that
    lands on the cap and area-averages onto it: the same kind of mean over a
    non-integer footprint, spending the budget instead of overshooting it
    (recognize-decode-bound.md §7 — the overshoot truncated a crop's text)."""
    w, h = crop.width, crop.height
    if cap <= 0 or w * h <= cap:
        return crop
    if mode != "box":
        while crop.width * crop.height > cap and crop.width >= 2 and crop.height >= 2:
            crop = crop.reduce(2)
        return crop
    scale = (cap / (w * h)) ** 0.5
    return crop.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BOX)


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
    # The device this engine actually loaded onto (pick_device of the override or
    # DEFAULT_DEVICE), recorded by load() so inference reads the resolved device
    # instead of each subclass tracking it. Class default keeps __init__()-skipping
    # subclasses safe, same as _loaded/_device_override.
    _device: str = "cpu"

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
        self._device = device
        self._loaded = True
        # Uniform load line for every local engine (per-engine logger namespace kept).
        self._log.info("%s loaded on %s", self.display_name, device_label(device))

    def unload(self) -> None:
        self._unload()
        self._loaded = False
        release_cuda_cache()
