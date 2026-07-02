"""mangaocr test suite: ``python -m tests`` (from packages/scanlation-mangaocr/).
The only test is a slow smoke that self-skips without the model weights."""
from __future__ import annotations

import sys

from scanlation_sdk.testing import run_modules
from tests import test_mangaocr

if __name__ == "__main__":
    sys.exit(run_modules([test_mangaocr]))
