"""llama.cpp test suite: ``python -m tests`` (from packages/scanlation-llama-cpp/).
Unit tests with the HTTP call mocked — no live server needed."""
from __future__ import annotations

import sys

from scanlation_sdk.testing import run_modules
from tests import test_llama_cpp

if __name__ == "__main__":
    sys.exit(run_modules([test_llama_cpp]))
