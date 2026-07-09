"""manga-ocr smoke test (slow). Self-skips unless the package AND its model
weights are present. Run on its own:

    python -m tests   (from packages/scanlation-manga-ocr/)

The smoke body is shared (scanlation_sdk.testing.recognizer_smoke) — only the
engine class, the availability probe, and the two SKIP strings differ.
"""
from __future__ import annotations

from scanlation_manga_ocr.plugin import MangaOcrRecognizer
from scanlation_sdk.testing import recognizer_smoke

TESTS = [
    recognizer_smoke(
        MangaOcrRecognizer,
        "manga_ocr",
        "SKIP: manga-ocr package not installed",
        "SKIP: manga-ocr model weights not downloaded",
    ),
]

if __name__ == "__main__":
    import sys

    from scanlation_sdk.testing import run

    sys.exit(run(TESTS, "test_manga_ocr (slow)"))
