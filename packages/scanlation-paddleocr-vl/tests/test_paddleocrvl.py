"""PaddleOCR-VL smoke test (slow). Self-skips unless transformers AND the model
weights are present. Run on its own:

    python -m tests   (from packages/scanlation-paddleocr-vl/)
"""
from __future__ import annotations

import importlib.util

from PIL import Image


def test_paddleocrvl_recognize_returns_str():
    if importlib.util.find_spec("transformers") is None:
        return "SKIP: transformers not installed"
    from scanlation_paddleocr_vl.plugin import PaddleOcrVLRecognizer
    from scanlation_sdk.contracts import Region

    rec = PaddleOcrVLRecognizer()
    if not rec.is_installed():
        return "SKIP: PaddleOCR-VL weights not downloaded"
    rec.load()
    out = rec.recognize(Image.new("RGB", (160, 64), (255, 255, 255)), Region.from_bbox(0, 0, 160, 64), {})
    assert isinstance(out, str)


TESTS = [test_paddleocrvl_recognize_returns_str]

if __name__ == "__main__":
    import sys

    from scanlation_sdk.testing import run

    sys.exit(run(TESTS, "test_paddleocrvl (slow)"))
