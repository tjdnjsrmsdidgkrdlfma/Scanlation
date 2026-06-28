"""Decode a comic-text-detector segmentation mask into rotated text Regions.

The detection bottleneck for manga is getting *clean rotated quads*. CTD's
segmentation head gives a per-pixel text mask; turning that mask into rotated
line quads is model-agnostic and exactly what feeds deskew:

    threshold -> contours -> (optional unclip dilation) -> minAreaRect -> quad
    -> map back through the letterbox transform to original-image pixels.

Kept separate from plugin.py so the geometry is unit-testable without onnxruntime.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.contracts import Region


def letterbox(img: np.ndarray, new_size: int, pad_value: int = 114):
    """Resize keeping aspect ratio, pad to a square new_size. YOLO-style.

    Returns (padded, ratio, (pad_w, pad_h)) so quads can be mapped back.
    """
    h, w = img.shape[:2]
    ratio = min(new_size / h, new_size / w)
    nh, nw = int(round(h * ratio)), int(round(w * ratio))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = (new_size - nw) // 2, (new_size - nh) // 2
    out = np.full((new_size, new_size, img.shape[2]), pad_value, dtype=img.dtype)
    out[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
    return out, ratio, (pad_w, pad_h)


def _unclip(quad: np.ndarray, ratio: float) -> np.ndarray:
    """Expand a quad outward by `ratio` (text masks tend to under-cover glyphs).

    Uses pyclipper only (no shapely); always re-fits to a 4-point quad so
    Region.from_quad stays happy.
    """
    try:
        import pyclipper
    except ImportError:
        return quad
    q = quad.astype(np.float32)
    area = abs(cv2.contourArea(q))
    length = cv2.arcLength(q, True)
    if length < 1e-6 or area <= 0:
        return quad
    distance = area * ratio / length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(q.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    if not expanded:
        return quad
    rect = cv2.minAreaRect(np.array(expanded[0], dtype=np.float32))
    return cv2.boxPoints(rect).astype(np.float32)


def mask_to_regions(
    mask: np.ndarray,
    ratio: float,
    pad: tuple[int, int],
    orig_w: int,
    orig_h: int,
    *,
    thresh: float = 0.3,
    min_area: int = 16,
    unclip_ratio: float = 1.2,
) -> list[Region]:
    """Convert a float mask (letterboxed coords) to original-pixel Regions."""
    pad_w, pad_h = pad
    binary = (mask >= thresh).astype(np.uint8) * 255
    if binary.ndim == 3:
        binary = binary[..., 0]

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[Region] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        rect = cv2.minAreaRect(cnt)               # ((cx,cy),(w,h),angle)
        quad = cv2.boxPoints(rect).astype(np.float32)
        if unclip_ratio and unclip_ratio > 0:
            quad = _unclip(quad, unclip_ratio - 1.0)

        # map letterboxed -> original pixels
        quad = quad.copy()
        quad[:, 0] = np.clip((quad[:, 0] - pad_w) / ratio, 0, orig_w - 1)
        quad[:, 1] = np.clip((quad[:, 1] - pad_h) / ratio, 0, orig_h - 1)

        (_, _), (rw, rh), angle = rect
        vertical = rh > rw  # taller than wide -> likely vertical JP line
        regions.append(Region.from_quad(quad, angle=float(angle), vertical=bool(vertical)))
    return regions
