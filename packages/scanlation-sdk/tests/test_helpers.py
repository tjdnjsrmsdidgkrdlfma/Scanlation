"""Unit tests for the shared plugin helpers: to_rgb, install_hint, EngineBase._log.

These are what R6 hoisted out of the local-model plugins; testing them here means
they're covered without weights (the engine smokes skip when weights are absent).
"""
from __future__ import annotations

from PIL import Image

import os

from scanlation_sdk import EngineBase, install_hint, to_rgb
from scanlation_sdk.http_translator import http_timeout


def test_to_rgb_passes_through_rgb():
    """An already-RGB image is returned as-is — no needless copy."""
    img = Image.new("RGB", (8, 8), (10, 20, 30))
    assert to_rgb(img) is img


def test_to_rgb_converts_non_rgb():
    """A non-RGB image is converted to a fresh RGB image."""
    for mode in ("L", "RGBA", "P"):
        out = to_rgb(Image.new(mode, (8, 8)))
        assert out.mode == "RGB"


def test_install_hint_default_matches_template():
    """The default hint names the engine in both the JSON key and the command,
    and ends in a period. (Byte-locks the format the plugins depend on.)"""
    assert install_hint("manga-ocr") == (
        'Install first: POST /install_plugins/ {"manga-ocr": true}, or '
        "`python tools/install.py manga-ocr`."
    )


def test_install_hint_extra_replaces_period():
    """`extra` replaces the trailing period with an engine-specific clause."""
    hint = install_hint(
        "comic-text-and-bubble-detector",
        extra=", or set SCANLATION_COMIC_TEXT_AND_BUBBLE_DETECTOR_MODEL=/path/to/model_dir.",
    )
    assert hint == (
        'Install first: POST /install_plugins/ {"comic-text-and-bubble-detector": true}, or '
        "`python tools/install.py comic-text-and-bubble-detector`, "
        "or set SCANLATION_COMIC_TEXT_AND_BUBBLE_DETECTOR_MODEL=/path/to/model_dir."
    )
    assert not hint.endswith("`.")  # the bare-period terminator is gone


def test_engine_base_log_is_namespaced():
    """EngineBase._log is a logger named scanlation.<name>, shared by every plugin."""
    class _Probe(EngineBase):
        name = "probe-engine"

    assert _Probe()._log.name == "scanlation.probe-engine"


def test_http_timeout_default_and_env():
    """The LLM HTTP client timeout is 10.0s by default, overridable via env."""
    os.environ.pop("SCANLATION_HTTP_TIMEOUT", None)
    assert http_timeout() == 10.0
    os.environ["SCANLATION_HTTP_TIMEOUT"] = "3.5"
    try:
        assert http_timeout() == 3.5
    finally:
        os.environ.pop("SCANLATION_HTTP_TIMEOUT", None)


TESTS = [
    test_to_rgb_passes_through_rgb,
    test_to_rgb_converts_non_rgb,
    test_install_hint_default_matches_template,
    test_install_hint_extra_replaces_period,
    test_engine_base_log_is_namespaced,
    test_http_timeout_default_and_env,
]

if __name__ == "__main__":
    import sys

    from scanlation_sdk.testing import run

    sys.exit(run(TESTS, "test_helpers"))
