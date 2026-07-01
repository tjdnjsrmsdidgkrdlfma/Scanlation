"""Fast test suite: ``python -m tests`` (from server/).

Runs every model-free module. The slow model smokes (test_ctd, test_mangaocr)
are run on their own: ``python -m tests.test_ctd`` — they self-skip when the
weights/package aren't present.
"""
from __future__ import annotations

import sys

from tests import (
    test_contracts,
    test_geometry,
    test_llamacpp,
    test_ollama,
    test_pipeline,
    test_routes,
)
from tests.helpers import run

MODULES = [
    test_contracts,
    test_geometry,
    test_pipeline,
    test_routes,
    test_ollama,
    test_llamacpp,
]


def main() -> int:
    rc = 0
    for mod in MODULES:
        rc |= run(mod.TESTS, mod.__name__)
    return rc


if __name__ == "__main__":
    sys.exit(main())
