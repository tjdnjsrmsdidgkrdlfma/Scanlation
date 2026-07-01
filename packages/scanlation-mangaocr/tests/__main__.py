"""mangaocr test suite: ``python -m tests`` (from packages/scanlation-mangaocr/).
The only test is a slow smoke that self-skips without the model weights."""
from __future__ import annotations

import sys

from scanlation_sdk.testing import run
from tests import test_mangaocr

MODULES = [test_mangaocr]


def main() -> int:
    rc = 0
    for mod in MODULES:
        rc |= run(mod.TESTS, mod.__name__)
    return rc


if __name__ == "__main__":
    sys.exit(main())
