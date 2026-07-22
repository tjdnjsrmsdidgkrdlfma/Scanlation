"""Deskew: warp a (possibly rotated) detected quad to an upright crop.

Independent reimplementation from standard OpenCV primitives
(order corners -> getPerspectiveTransform -> warpPerspective). No code is
copied from manga-image-translator (GPLv3); only the well-known homography
recipe is used.

Axis-aligned quads take a PIL-crop fast path (exact, and avoids importing cv2),
so the dummy/no-rotation pipeline runs even where OpenCV is unavailable.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from scanlation_sdk.contracts import Region

# Axis-aligned tolerance (px): a detected quad whose edges sit within this of a
# rectangle takes the exact PIL-crop fast path instead of a perspective warp.
_AXIS_ALIGNED_EPS = 1.0
# Minimum crop side (px): tiny crops are padded up to this so the recognizer gets
# something usable. Internal geometric heuristics, kept as named constants rather
# than /admin fields (like idle_unload's sweep cadence).
_MIN_CROP_SIZE = 8


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL (clockwise from top-left).

    Uses the classic sum/diff trick: TL has min x+y, BR has max x+y, TR has
    min (y-x), BL has max (y-x).
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _target_size(ordered: np.ndarray) -> tuple[int, int]:
    tl, tr, br, bl = ordered
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    return int(round(width)), int(round(height))


def _is_axis_aligned(ordered: np.ndarray, eps: float = _AXIS_ALIGNED_EPS) -> bool:
    tl, tr, br, bl = ordered
    return (
        abs(tl[1] - tr[1]) <= eps and abs(bl[1] - br[1]) <= eps and
        abs(tl[0] - bl[0]) <= eps and abs(tr[0] - br[0]) <= eps
    )


def deskew_crop(img: Image.Image, region: Region, min_size: int = _MIN_CROP_SIZE) -> Image.Image:
    """Return an upright RGB crop for ``region``.

    Vertical Japanese text is intentionally left vertical (manga-ocr reads
    vertical natively); only the skew is removed, not the writing direction.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    ordered = order_quad(region.polygon)
    w, h = _target_size(ordered)
    if w < 1 or h < 1:
        # Zero-area quad: hand back a tiny blank; the pipeline skips empties.
        return Image.new("RGB", (max(w, 1), max(h, 1)), (255, 255, 255))

    # Fast path: axis-aligned rectangle -> exact PIL crop, no warp/cv2.
    if _is_axis_aligned(ordered):
        x0, y0, x1, y1 = region.bbox
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img.width, x1), min(img.height, y1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            return Image.new("RGB", (min_size, min_size), (255, 255, 255))
        crop = img.crop((x0, y0, x1, y1))
    else:
        import cv2  # lazy: only rotated quads need OpenCV

        src = np.ascontiguousarray(ordered, dtype=np.float32)
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            np.asarray(img), matrix, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        crop = Image.fromarray(warped)

    # Pad up tiny crops so the recognizer gets something usable.
    if crop.width < min_size or crop.height < min_size:
        padded = Image.new("RGB", (max(crop.width, min_size), max(crop.height, min_size)), (255, 255, 255))
        padded.paste(crop, (0, 0))
        crop = padded
    return crop
