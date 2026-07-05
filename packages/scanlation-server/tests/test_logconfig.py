"""app.logconfig smoke — configure_logging wires levels without error.

Runs last in the suite (it reconfigures global logging). Model/GPU free.
"""
from __future__ import annotations

import logging

from app.logconfig import configure_logging
from tests.helpers import run


def test_configure_logging_sets_levels():
    configure_logging("DEBUG")
    # our namespace opens to the requested level; children (engine plugin loggers) inherit it
    assert logging.getLogger("scanlation").getEffectiveLevel() == logging.DEBUG
    assert logging.getLogger("scanlation.comic-text-and-bubble-detector").getEffectiveLevel() == logging.DEBUG
    # third-party stays gated at the root WARNING (transformers/httpx don't drown the log)
    assert logging.getLogger().getEffectiveLevel() == logging.WARNING
    configure_logging("INFO")  # restore a sane default
    assert logging.getLogger("scanlation").getEffectiveLevel() == logging.INFO


TESTS = [test_configure_logging_sets_levels]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_logconfig"))
