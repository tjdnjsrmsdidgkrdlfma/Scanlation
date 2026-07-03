"""RT-DETR smoke test (slow). Skipped unless transformers+torch AND the weights
are present. Install the weights first (tools/install.py rtdetr or POST
/install_plugins/ {"rtdetr": true}); not part of the fast model-free suite.

    python -m tests.test_rtdetr
"""
from __future__ import annotations

import importlib.util

from PIL import Image


def _available() -> bool:
    if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
        return False
    from scanlation_rtdetr.plugin import RTDetrDetector

    return RTDetrDetector().is_installed()


def test_rtdetr_runs_without_crashing():
    if not _available():
        return "SKIP: transformers/torch or rtdetr weights not present"
    from scanlation_rtdetr.plugin import RTDetrDetector

    detector = RTDetrDetector()
    detector.load()
    img = Image.new("RGB", (800, 1200), (255, 255, 255))
    regions = detector.detect(img, {})
    assert isinstance(regions, list)
    for r in regions:
        assert r.polygon.shape == (4, 2)


TESTS = [test_rtdetr_runs_without_crashing]

if __name__ == "__main__":
    import sys

    from scanlation_sdk.testing import run

    sys.exit(run(TESTS, "test_rtdetr (slow)"))
