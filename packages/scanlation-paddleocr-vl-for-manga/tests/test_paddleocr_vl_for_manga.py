"""PaddleOCR-VL smoke test (slow). Self-skips unless transformers AND the model
weights are present. Run on its own:

    python -m tests   (from packages/scanlation-paddleocr-vl-for-manga/)

The smoke body is shared (scanlation_sdk.testing.recognizer_smoke) — only the
engine class, the availability probe, and the two SKIP strings differ.
"""
from __future__ import annotations

from scanlation_paddleocr_vl_for_manga.plugin import PaddleOcrVLForMangaRecognizer
from scanlation_sdk.testing import recognizer_smoke


def test_downscale_options_default_on():
    """The resolution cap ships on: max_pixels defaults to a positive int, mode to pow2,
    and resolve_options fills both for a bare call. Fast — no weights needed."""
    schema = PaddleOcrVLForMangaRecognizer.OPTION_SCHEMA
    assert isinstance(schema["max_pixels"]["default"], int) and schema["max_pixels"]["default"] > 0
    assert schema["downscale_mode"]["default"] == "pow2"
    resolved = PaddleOcrVLForMangaRecognizer().resolve_options({})
    assert resolved["max_pixels"] == schema["max_pixels"]["default"]
    assert resolved["downscale_mode"] == "pow2"


TESTS = [
    test_downscale_options_default_on,
    recognizer_smoke(
        PaddleOcrVLForMangaRecognizer,
        "transformers",
        "SKIP: transformers not installed",
        "SKIP: PaddleOCR-VL weights not downloaded",
    ),
]

if __name__ == "__main__":
    import sys

    from scanlation_sdk.testing import run

    sys.exit(run(TESTS, "test_paddleocr_vl_for_manga (slow)"))
