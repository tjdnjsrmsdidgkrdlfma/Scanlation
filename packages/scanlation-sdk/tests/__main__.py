"""`python -m tests` for scanlation-sdk."""
from __future__ import annotations

import sys

from scanlation_sdk.testing import run_modules
from tests import test_helpers

if __name__ == "__main__":
    sys.exit(run_modules([test_helpers]))
