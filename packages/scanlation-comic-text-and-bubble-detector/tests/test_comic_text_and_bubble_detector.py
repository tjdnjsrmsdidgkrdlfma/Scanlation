"""comic-text-and-bubble-detector smoke test (slow). Skipped unless
transformers+torch AND the weights are present. Install the weights first
(tools/install.py comic-text-and-bubble-detector or POST /install_plugins/
{"comic-text-and-bubble-detector": true}); not part of the fast model-free suite.

    python -m tests.test_comic_text_and_bubble_detector
"""
from __future__ import annotations

import importlib.util

from PIL import Image


def _available() -> bool:
    if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
        return False
    from scanlation_comic_text_and_bubble_detector.plugin import ComicTextAndBubbleDetector

    return ComicTextAndBubbleDetector().is_installed()


def test_comic_text_and_bubble_detector_runs_without_crashing():
    if not _available():
        return "SKIP: transformers/torch or comic-text-and-bubble-detector weights not present"
    from scanlation_comic_text_and_bubble_detector.plugin import ComicTextAndBubbleDetector

    detector = ComicTextAndBubbleDetector()
    detector.load()
    img = Image.new("RGB", (800, 1200), (255, 255, 255))
    regions = detector.detect(img, {})
    assert isinstance(regions, list)
    for r in regions:
        assert r.polygon.shape == (4, 2)


TESTS = [test_comic_text_and_bubble_detector_runs_without_crashing]

if __name__ == "__main__":
    import sys

    from scanlation_sdk.testing import run

    sys.exit(run(TESTS, "test_comic_text_and_bubble_detector (slow)"))
