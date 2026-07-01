"""manga-ocr smoke test (slow). Skipped unless the package is installed. Not
part of the fast suite; run on its own:

    python -m tests.test_mangaocr
"""
from __future__ import annotations

import importlib.util

from PIL import Image


def test_mangaocr_recognize_returns_str():
    if importlib.util.find_spec("manga_ocr") is None:
        return "SKIP: manga-ocr not installed"
    from app.contracts import Region
    from app.registry import registry

    rec = registry.get("recognizer", "mangaocr")
    out = rec.recognize(Image.new("RGB", (160, 64), (255, 255, 255)), Region.from_bbox(0, 0, 160, 64), {})
    assert isinstance(out, str)


TESTS = [test_mangaocr_recognize_returns_str]

if __name__ == "__main__":
    import sys

    from tests.helpers import run

    sys.exit(run(TESTS, "test_mangaocr (slow)"))
