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

from scanlation_sdk.contracts import Region

# Single source of truth for the mask-decode tuning defaults. The plugin's
# OPTION_SCHEMA defaults, its detect() fallbacks, AND mask_to_regions()'s own
# keyword defaults all read these — so a standalone decode call (e.g. the tests)
# behaves exactly like the live pipeline instead of silently disabling merging.
DEFAULTS = {
    "mask_threshold": 0.3,   # decode calls this `thresh`
    "min_area": 200,
    "min_side": 12,
    "unclip_ratio": 1.2,
    "merge_px": 16,
    "merge_aspect": 1.7,
}


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
    thresh: float = DEFAULTS["mask_threshold"],
    min_area: int = DEFAULTS["min_area"],
    min_side: int = DEFAULTS["min_side"],
    unclip_ratio: float = DEFAULTS["unclip_ratio"],
    merge_px: int = DEFAULTS["merge_px"],
    merge_aspect: float = DEFAULTS["merge_aspect"],
) -> list[Region]:
    """Convert a float mask (letterboxed coords) to original-pixel Regions.

    ``merge_px`` morphologically closes the binary mask first, bridging the
    gaps between adjacent glyphs so a text line/bubble becomes ONE region
    instead of one-region-per-character (manga-ocr needs whole lines, not
    single characters). 0 disables merging. ``merge_aspect`` makes the close
    kernel taller than wide (>1) so glyphs bridge *along* a vertical JP column
    without also fusing neighbouring columns into one blob.

    Noise (SFX shards, ellipses, stray hearts) is dropped by two floors applied
    in **original pixels** (resolution-independent, unlike the old det-grid
    area): ``min_side`` on the rotated rect's short side — the single best
    discriminator, since real text lines are never hairline-thin — and
    ``min_area`` as an area backstop.
    """
    pad_w, pad_h = pad
    binary = (mask >= thresh).astype(np.uint8) * 255
    if binary.ndim == 3:
        binary = binary[..., 0]

    if merge_px and merge_px > 0:
        kw = max(1, int(round(merge_px)))
        kh = max(1, int(round(merge_px * (merge_aspect if merge_aspect > 0 else 1.0))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    inv = 1.0 / ratio if ratio else 1.0
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[Region] = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)               # ((cx,cy),(w,h),angle)
        (_, _), (rw, rh), angle = rect
        # rect dims in original pixels (ratio scales; letterbox pad only offsets)
        ow, oh = rw * inv, rh * inv
        if min(ow, oh) < min_side or (ow * oh) < min_area:
            continue
        quad = cv2.boxPoints(rect).astype(np.float32)
        if unclip_ratio and unclip_ratio > 0:
            # unclip_ratio is "1.0 = no expansion"; _unclip wants the extra fraction.
            quad = _unclip(quad, unclip_ratio - 1.0)

        # map letterboxed -> original pixels
        quad = quad.copy()
        quad[:, 0] = np.clip((quad[:, 0] - pad_w) * inv, 0, orig_w - 1)
        quad[:, 1] = np.clip((quad[:, 1] - pad_h) * inv, 0, orig_h - 1)

        vertical = rh > rw  # taller than wide -> likely vertical JP line
        regions.append(Region.from_quad(quad, angle=float(angle), vertical=bool(vertical)))
    return regions
