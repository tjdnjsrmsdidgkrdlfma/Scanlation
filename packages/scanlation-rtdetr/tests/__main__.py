"""rtdetr test suite: ``python -m tests`` (from packages/scanlation-rtdetr/).

test_rtdetr_postprocess (geometry) is model-free; test_rtdetr is a slow smoke
that self-skips unless the transformers weights + torch are present. The device
pick moved to the SDK (tested in the server's test_contracts).
"""
from __future__ import annotations

import sys

from scanlation_sdk.testing import run_modules
from tests import test_rtdetr, test_rtdetr_postprocess

if __name__ == "__main__":
    sys.exit(run_modules([test_rtdetr_postprocess, test_rtdetr]))
