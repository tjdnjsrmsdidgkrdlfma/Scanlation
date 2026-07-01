"""llamacpp test suite: ``python -m tests`` (from packages/scanlation-llamacpp/).
Unit tests with the HTTP call mocked — no live server needed."""
from __future__ import annotations

import sys

from scanlation_sdk.testing import run
from tests import test_llamacpp

MODULES = [test_llamacpp]


def main() -> int:
    rc = 0
    for mod in MODULES:
        rc |= run(mod.TESTS, mod.__name__)
    return rc


if __name__ == "__main__":
    sys.exit(main())
