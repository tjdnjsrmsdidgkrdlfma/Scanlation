"""PaddleOCR-VL-For-Manga test suite: ``python -m tests`` (from packages/scanlation-paddleocr-vl-for-manga/).
The only test is a slow smoke that self-skips without the model weights."""
from __future__ import annotations

import sys

from scanlation_sdk.testing import run_modules
from tests import test_paddleocr_vl_for_manga

if __name__ == "__main__":
    sys.exit(run_modules([test_paddleocr_vl_for_manga]))
