"""Deskew unit tests — the highest-risk math, covered with no models."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from scanlation_sdk.contracts import Region
from app.geometry import deskew_crop, order_quad


def test_order_quad_orders_tl_tr_br_bl():
    # deliberately shuffled
    pts = np.array([[30, 10], [0, 20], [10, 0], [20, 30]], dtype=np.float32)
    tl, tr, br, bl = order_quad(pts)
    assert tuple(tl) == (10, 0)
    assert tuple(tr) == (30, 10)
    assert tuple(br) == (20, 30)
    assert tuple(bl) == (0, 20)


def test_axis_aligned_deskew_is_exact_crop():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([50, 60, 150, 110], fill=(0, 0, 0))
    region = Region.from_bbox(50, 60, 150, 110)
    crop = deskew_crop(img, region)
    assert crop.size == (100, 50)
    # interior is the black fill
    assert crop.getpixel((50, 25)) == (0, 0, 0)


def _rotated_quad(cx, cy, w, h, deg):
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    base = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in base]


def test_rotated_deskew_recovers_upright_size_and_content():
    img = Image.new("RGB", (240, 240), (255, 255, 255))
    quad = _rotated_quad(120, 120, 100, 40, 30)
    ImageDraw.Draw(img).polygon(quad, fill=(0, 0, 0))
    region = Region.from_quad(quad, angle=30.0)

    crop = deskew_crop(img, region)
    # upright dimensions recovered within rounding/interpolation tolerance
    assert abs(crop.width - 100) <= 3
    assert abs(crop.height - 40) <= 3
    # the warped region was a solid fill -> overwhelmingly black
    arr = np.asarray(crop)
    black_frac = float((arr.sum(axis=2) < 60).mean())
    assert black_frac > 0.85


def test_zero_area_quad_is_safe():
    img = Image.new("RGB", (50, 50), (255, 255, 255))
    region = Region.from_quad([[10, 10], [10, 10], [10, 10], [10, 10]])
    crop = deskew_crop(img, region)
    assert crop.width >= 1 and crop.height >= 1


TESTS = [
    test_order_quad_orders_tl_tr_br_bl,
    test_axis_aligned_deskew_is_exact_crop,
    test_rotated_deskew_recovers_upright_size_and_content,
    test_zero_area_quad_is_safe,
]

if __name__ == "__main__":
    import sys

    from tests.helpers import run

    sys.exit(run(TESTS, "test_geometry"))
