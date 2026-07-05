"""comic-text-and-bubble-detector test suite: ``python -m tests`` (from packages/scanlation-comic-text-and-bubble-detector/).

test_comic_text_and_bubble_detector_postprocess (geometry) is model-free; test_comic_text_and_bubble_detector is a slow smoke
that self-skips unless the transformers weights + torch are present. The device
pick moved to the SDK (tested in the server's test_contracts).
"""
from __future__ import annotations

import sys

from scanlation_sdk.testing import run_modules
from tests import test_comic_text_and_bubble_detector, test_comic_text_and_bubble_detector_postprocess

if __name__ == "__main__":
    sys.exit(run_modules([test_comic_text_and_bubble_detector_postprocess, test_comic_text_and_bubble_detector]))
