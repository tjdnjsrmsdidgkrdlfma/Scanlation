"""CTDDetector — comic-text-detector (ONNX) text-region detector.

Detection is the real accuracy bottleneck for manga, so this plugin is the one
to verify visually (tools/visualize.py) once the ONNX weights are present.

Design choices that reduce the "unknown ONNX I/O" risk flagged in the design:
  * onnxruntime is imported lazily inside load(), so the module imports (and the
    registry discovers the class) with no native deps.
  * The segmentation-mask output is found by SHAPE (the 4-D output with the
    largest spatial area), not by a hardcoded tensor name — robust across the
    several comic-text-detector ONNX exports.
  * Provider selection follows SCANLATION_DEVICE (rocm/dml/cpu) but ALWAYS
    appends CPUExecutionProvider as a fallback, and logs the active provider.

Weights are NOT bundled. Set SCANLATION_CTD_MODEL=/path/to/model.onnx or drop an
.onnx into <models_dir>/ctd/. Until then load() raises a clear instruction and
the slow CTD test is skipped.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config import settings
from app.contracts import EngineBase, Region
from . import decode

logger = logging.getLogger("scanlation.ctd")

_PROVIDERS = {
    "rocm": "ROCMExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


class CTDDetector(EngineBase):
    name = "ctd"
    display_name = "comic-text-detector (ONNX)"
    homepage = "https://github.com/dmMaze/comic-text-detector"
    description = "Manga/comic text detector. Segmentation mask -> rotated line quads."
    warning = "Requires an ONNX weight file (SCANLATION_CTD_MODEL or <models>/ctd/*.onnx)."
    OPTION_SCHEMA = {
        "det_size": {"type": int, "default": 1024, "description": "Square inference size (letterboxed)."},
        "mask_threshold": {"type": float, "default": 0.3, "description": "Mask binarization threshold."},
        "min_area": {"type": int, "default": 16, "description": "Drop mask blobs smaller than this (px^2)."},
        "unclip_ratio": {"type": float, "default": 1.2, "description": "Dilate quads outward (1.0 = none)."},
    }
    SUPPORTED_SRC = ["ja", "en", "zh", "ko"]

    def __init__(self) -> None:
        self._session = None
        self._input_name: str | None = None
        self._det_size = 1024

    # --- weights ---
    def _resolve_model_path(self) -> Path:
        import os

        env = os.environ.get("SCANLATION_CTD_MODEL")
        if env:
            p = Path(env)
            if p.is_file():
                return p
        ctd_dir = settings.models_dir / "ctd"
        if ctd_dir.is_dir():
            onnx = sorted(ctd_dir.glob("*.onnx"))
            if onnx:
                return onnx[0]
        raise RuntimeError(
            "CTD ONNX weights not found. Set SCANLATION_CTD_MODEL=/path/model.onnx or place "
            f"an .onnx in {ctd_dir}. See e.g. huggingface 'mayocream/comic-text-detector-onnx'."
        )

    def _providers(self) -> list[str]:
        preferred = _PROVIDERS.get(settings.device.lower())
        order = []
        if preferred:
            order.append(preferred)
        order.append("CPUExecutionProvider")  # always a fallback
        return order

    def load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort  # lazy

        model_path = self._resolve_model_path()
        available = set(ort.get_available_providers())
        providers = [p for p in self._providers() if p in available] or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        # static input size if the export pins it, else keep the option default
        shape = self._session.get_inputs()[0].shape
        if isinstance(shape[-1], int) and shape[-1] > 0:
            self._det_size = int(shape[-1])
        logger.info("CTD loaded %s providers=%s det_size=%d", model_path.name,
                    self._session.get_providers(), self._det_size)

    def unload(self) -> None:
        self._session = None

    # --- inference ---
    @staticmethod
    def _pick_mask(outputs: list[np.ndarray]) -> np.ndarray:
        """Pick the segmentation output: the 4-D tensor with the largest H*W."""
        best, best_area = None, -1
        for o in outputs:
            if o.ndim == 4:
                area = o.shape[-1] * o.shape[-2]
                if area > best_area:
                    best, best_area = o, area
        if best is None:
            raise RuntimeError(f"No 4-D mask output found; got shapes {[o.shape for o in outputs]}")
        mask = best[0]
        mask = mask[0] if mask.shape[0] <= 4 else mask  # (C,H,W) -> text channel 0
        if mask.min() < 0.0 or mask.max() > 1.0:        # logits -> sigmoid
            mask = 1.0 / (1.0 + np.exp(-mask))
        return mask.astype(np.float32)

    def detect(self, image: Image.Image, options: dict[str, Any]) -> list[Region]:
        if self._session is None:
            self.load()
        det_size = int(options.get("det_size", self._det_size))
        img = np.asarray(image.convert("RGB"))
        orig_h, orig_w = img.shape[:2]

        padded, ratio, pad = decode.letterbox(img, det_size)
        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]  # NCHW

        outputs = self._session.run(None, {self._input_name: blob})
        mask = self._pick_mask(outputs)

        # mask is at the network's spatial resolution; rescale to det_size grid
        if mask.shape[:2] != (det_size, det_size):
            import cv2

            mask = cv2.resize(mask, (det_size, det_size), interpolation=cv2.INTER_LINEAR)

        return decode.mask_to_regions(
            mask, ratio, pad, orig_w, orig_h,
            thresh=float(options.get("mask_threshold", 0.3)),
            min_area=int(options.get("min_area", 16)),
            unclip_ratio=float(options.get("unclip_ratio", 1.2)),
        )
