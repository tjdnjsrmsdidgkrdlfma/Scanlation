"""ctd test suite: ``python -m tests`` (from packages/scanlation-ctd/).

test_ctd_decode is model-free (synthetic mask); test_ctd is a slow smoke that
self-skips unless ONNX weights are present.
"""
from __future__ import annotations

import sys

from scanlation_sdk.testing import run_modules
from tests import test_ctd, test_ctd_decode

if __name__ == "__main__":
    sys.exit(run_modules([test_ctd_decode, test_ctd]))
