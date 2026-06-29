"""manga-ocr smoke test (slow). Skipped unless the package is installed."""
from __future__ import annotations

import importlib.util

import pytest
from PIL import Image

pytestmark = pytest.mark.slow


@pytest.mark.skipif(importlib.util.find_spec("manga_ocr") is None, reason="manga-ocr not installed")
def test_mangaocr_recognize_returns_str():
    from app.contracts import Region
    from app.registry import registry

    rec = registry.get("recognizer", "mangaocr")
    out = rec.recognize(Image.new("RGB", (160, 64), (255, 255, 255)), Region.from_bbox(0, 0, 160, 64), {})
    assert isinstance(out, str)
