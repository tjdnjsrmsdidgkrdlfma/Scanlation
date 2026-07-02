"""Fast test suite: ``python -m tests`` (from server/).

Runs every model-free module. The slow model smokes live in each engine package
(``packages/scanlation-<engine>/tests/``); run ``python -m tests`` from there —
they self-skip when the weights/package aren't present.
"""
from __future__ import annotations

import sys

from tests import (
    test_contracts,
    test_geometry,
    test_pipeline,
    test_routes,
)
from tests.helpers import run_modules

if __name__ == "__main__":
    sys.exit(run_modules([test_contracts, test_geometry, test_pipeline, test_routes]))
