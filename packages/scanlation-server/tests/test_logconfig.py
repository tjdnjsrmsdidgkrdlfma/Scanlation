"""app.logconfig — level wiring and the verbose toggle.

Runs last in the suite (it reconfigures global logging). Model/GPU free.
"""
from __future__ import annotations

import logging

from app.logconfig import apply_verbose, configure_logging
from tests.helpers import run


def _level() -> int:
    return logging.getLogger("scanlation").getEffectiveLevel()


def test_configure_logging_sets_levels():
    configure_logging("DEBUG")
    # our namespace opens to the requested level; children (engine plugin loggers) inherit it
    assert logging.getLogger("scanlation").getEffectiveLevel() == logging.DEBUG
    assert logging.getLogger("scanlation.comic-text-and-bubble-detector").getEffectiveLevel() == logging.DEBUG
    # third-party stays gated at the root WARNING (transformers/httpx don't drown the log)
    assert logging.getLogger().getEffectiveLevel() == logging.WARNING
    configure_logging("INFO")  # restore a sane default
    assert logging.getLogger("scanlation").getEffectiveLevel() == logging.INFO


def test_verbose_off_returns_to_the_configured_level():
    """The startup path: configure_logging(SCANLATION_LOG_LEVEL) then apply_verbose
    (from the persisted 동작 toggle). Turning verbose off must not open the log back
    up to INFO when the env asked for WARNING."""
    try:
        configure_logging("WARNING")
        assert _level() == logging.WARNING
        apply_verbose(False)          # what lifespan does when the toggle is off
        assert _level() == logging.WARNING
        apply_verbose(True)           # /admin turns it on
        assert _level() == logging.DEBUG
        apply_verbose(False)          # ...and off again -> back to WARNING, not INFO
        assert _level() == logging.WARNING
    finally:
        configure_logging("INFO")
        apply_verbose(False)


def test_verbose_off_does_not_downgrade_a_debug_env():
    """SCANLATION_LOG_LEVEL=DEBUG with the toggle off stays DEBUG — the env asked for it."""
    try:
        configure_logging("DEBUG")
        apply_verbose(False)
        assert _level() == logging.DEBUG
    finally:
        configure_logging("INFO")
        apply_verbose(False)


TESTS = [
    test_configure_logging_sets_levels,
    test_verbose_off_returns_to_the_configured_level,
    test_verbose_off_does_not_downgrade_a_debug_env,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_logconfig"))
