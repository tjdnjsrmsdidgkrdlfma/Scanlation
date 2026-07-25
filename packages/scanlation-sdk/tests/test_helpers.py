"""Unit tests for the shared plugin helpers: to_rgb, install_hint, EngineBase._log.

These are what R6 hoisted out of the local-model plugins; testing them here means
they're covered without weights (the engine smokes skip when weights are absent).
"""
from __future__ import annotations

from PIL import Image

import os

from scanlation_sdk import EngineBase, downscale_to_cap, install_hint, to_rgb
from scanlation_sdk.http_translator import http_timeout, list_timeout


def test_to_rgb_passes_through_rgb():
    """An already-RGB image is returned as-is — no needless copy."""
    img = Image.new("RGB", (8, 8), (10, 20, 30))
    assert to_rgb(img) is img


def test_to_rgb_converts_non_rgb():
    """A non-RGB image is converted to a fresh RGB image."""
    for mode in ("L", "RGBA", "P"):
        out = to_rgb(Image.new(mode, (8, 8)))
        assert out.mode == "RGB"


def test_downscale_to_cap_noop_when_off_or_small():
    """cap<=0, or a crop already under the cap, is returned unchanged (same object)."""
    img = Image.new("RGB", (400, 300))  # 120k px
    assert downscale_to_cap(img, 0, "pow2") is img       # off
    assert downscale_to_cap(img, -1, "pow2") is img      # off (negative)
    assert downscale_to_cap(img, 150000, "pow2") is img  # 120k <= 150k -> untouched


def test_downscale_to_cap_pow2_halves_until_under():
    """pow2 halves by exact 2x2 blocks until <= cap (overshoots well under)."""
    out = downscale_to_cap(Image.new("RGB", (1000, 1000)), 150000, "pow2")  # 1,000,000 px
    assert out.size == (250, 250)  # 1000 -> 500 (250k) -> 250 (62.5k <= 150k)
    assert out.width * out.height <= 150000


def test_downscale_to_cap_box_lands_at_cap_aspect_kept():
    """box shrinks to ~cap px with aspect ratio preserved — it SPENDS the budget
    where pow2 overshoots to a power-of-two fraction of it."""
    out = downscale_to_cap(Image.new("RGB", (1200, 800)), 150000, "box")  # 3:2
    assert 0.99 * 150000 <= out.width * out.height <= 150000
    assert abs(out.width / out.height - 1.5) < 0.02


def test_downscale_to_cap_unknown_mode_falls_back_to_pow2():
    """An unrecognized mode must not silently become `box`: pow2 is the safe default,
    and the admin only ever offers DOWNSCALE_MODES."""
    img = Image.new("RGB", (1000, 1000))
    assert downscale_to_cap(img, 150000, "grid28").size == downscale_to_cap(img, 150000, "pow2").size


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


def test_list_timeout_default_and_env():
    """The admin model-list probe timeout is 4.0s by default, overridable via env."""
    os.environ.pop("SCANLATION_HTTP_LIST_TIMEOUT", None)
    assert list_timeout() == 4.0
    os.environ["SCANLATION_HTTP_LIST_TIMEOUT"] = "1.5"
    try:
        assert list_timeout() == 1.5
    finally:
        os.environ.pop("SCANLATION_HTTP_LIST_TIMEOUT", None)


TESTS = [
    test_to_rgb_passes_through_rgb,
    test_to_rgb_converts_non_rgb,
    test_downscale_to_cap_noop_when_off_or_small,
    test_downscale_to_cap_pow2_halves_until_under,
    test_downscale_to_cap_box_lands_at_cap_aspect_kept,
    test_downscale_to_cap_unknown_mode_falls_back_to_pow2,
    test_install_hint_default_matches_template,
    test_install_hint_extra_replaces_period,
    test_engine_base_log_is_namespaced,
    test_http_timeout_default_and_env,
    test_list_timeout_default_and_env,
]

if __name__ == "__main__":
    import sys

    from scanlation_sdk.testing import run

    sys.exit(run(TESTS, "test_helpers"))
