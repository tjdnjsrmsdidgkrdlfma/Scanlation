"""manga-ocr test suite: ``python -m tests`` (from packages/scanlation-manga-ocr/).
The only test is a slow smoke that self-skips without the model weights."""
from __future__ import annotations

import sys

from scanlation_sdk.testing import run_modules
from tests import test_manga_ocr

if __name__ == "__main__":
    sys.exit(run_modules([test_manga_ocr]))
