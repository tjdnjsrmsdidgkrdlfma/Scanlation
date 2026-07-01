"""ctd test suite: ``python -m tests`` (from packages/scanlation-ctd/).

test_ctd_decode is model-free (synthetic mask); test_ctd is a slow smoke that
self-skips unless ONNX weights are present.
"""
from __future__ import annotations

import sys

from scanlation_sdk.testing import run
from tests import test_ctd, test_ctd_decode

MODULES = [test_ctd_decode, test_ctd]


def main() -> int:
    rc = 0
    for mod in MODULES:
        rc |= run(mod.TESTS, mod.__name__)
    return rc


if __name__ == "__main__":
    sys.exit(main())
