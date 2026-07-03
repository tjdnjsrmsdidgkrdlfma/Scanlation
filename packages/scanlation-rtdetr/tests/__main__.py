"""rtdetr test suite: ``python -m tests`` (from packages/scanlation-rtdetr/).

test_rtdetr_postprocess (geometry) and test_rtdetr_device (device pick) are
model-free; test_rtdetr is a slow smoke that self-skips unless the transformers
weights + torch are present.
"""
from __future__ import annotations

import sys

from scanlation_sdk.testing import run_modules
from tests import test_rtdetr, test_rtdetr_device, test_rtdetr_postprocess

if __name__ == "__main__":
    sys.exit(run_modules([test_rtdetr_postprocess, test_rtdetr_device, test_rtdetr]))
