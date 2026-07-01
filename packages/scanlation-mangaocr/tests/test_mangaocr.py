"""manga-ocr smoke test (slow). Self-skips unless the package AND its model
weights are present. Run on its own:

    python -m tests   (from packages/scanlation-mangaocr/)
"""
from __future__ import annotations

import importlib.util

from PIL import Image


def test_mangaocr_recognize_returns_str():
    if importlib.util.find_spec("manga_ocr") is None:
        return "SKIP: manga-ocr package not installed"
    from scanlation_mangaocr.plugin import MangaOcrRecognizer
    from scanlation_sdk.contracts import Region

    rec = MangaOcrRecognizer()
    if not rec.is_installed():
        return "SKIP: manga-ocr model weights not downloaded"
    rec.load()
    out = rec.recognize(Image.new("RGB", (160, 64), (255, 255, 255)), Region.from_bbox(0, 0, 160, 64), {})
    assert isinstance(out, str)


TESTS = [test_mangaocr_recognize_returns_str]

if __name__ == "__main__":
    import sys

    from scanlation_sdk.testing import run

    sys.exit(run(TESTS, "test_mangaocr (slow)"))
