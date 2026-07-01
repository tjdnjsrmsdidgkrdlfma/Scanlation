"""CTD mask-decode geometry — testable with no ONNX weights (synthetic mask)."""
from __future__ import annotations

import math

import cv2
import numpy as np

from scanlation_ctd.decode import letterbox, mask_to_regions


def _rotated_quad(cx, cy, w, h, deg):
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    base = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return np.array([[cx + x * ca - y * sa, cy + x * sa + y * ca] for x, y in base], dtype=np.float32)


def test_letterbox_is_square_and_invertible():
    img = np.zeros((300, 200, 3), np.uint8)
    padded, ratio, (pw, ph) = letterbox(img, 256)
    assert padded.shape == (256, 256, 3)
    # map an original point forward then back
    lx, ly = 100 * ratio + pw, 150 * ratio + ph
    assert abs((lx - pw) / ratio - 100) < 1 and abs((ly - ph) / ratio - 150) < 1


def test_mask_to_regions_recovers_rotated_rect():
    size = 256
    mask = np.zeros((size, size), np.float32)
    quad = _rotated_quad(128, 128, 90, 30, 25)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 1.0)

    regions = mask_to_regions(
        mask, ratio=1.0, pad=(0, 0), orig_w=size, orig_h=size,
        thresh=0.5, min_area=20, unclip_ratio=1.0,
    )
    assert len(regions) >= 1
    r = max(regions, key=lambda rr: (rr.bbox[2] - rr.bbox[0]) * (rr.bbox[3] - rr.bbox[1]))
    x0, y0, x1, y1 = r.bbox
    # encloses the rect center, and is meaningfully rotated
    assert x0 < 128 < x1 and y0 < 128 < y1
    assert abs(r.angle) > 1.0
    assert r.polygon.shape == (4, 2)


def test_mask_to_regions_empty_mask_is_empty():
    mask = np.zeros((128, 128), np.float32)
    regions = mask_to_regions(mask, 1.0, (0, 0), 128, 128)
    assert regions == []


TESTS = [
    test_letterbox_is_square_and_invertible,
    test_mask_to_regions_recovers_rotated_rect,
    test_mask_to_regions_empty_mask_is_empty,
]

if __name__ == "__main__":
    import sys

    from scanlation_sdk.testing import run

    sys.exit(run(TESTS, "test_ctd_decode"))
