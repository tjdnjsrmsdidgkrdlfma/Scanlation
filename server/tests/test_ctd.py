"""CTD smoke test (slow). Skipped unless ONNX weights are present.

Set SCANLATION_CTD_MODEL or drop an .onnx into <models>/ctd/. Provide a real
manga page via SCANLATION_CTD_FIXTURE to assert region count > 0. Not part of
the fast suite; run on its own (self-skips when weights are absent):

    python -m tests.test_ctd
"""
from __future__ import annotations

import os

from PIL import Image


def _model_available() -> bool:
    if os.environ.get("SCANLATION_CTD_MODEL"):
        return True
    from app.config import settings

    d = settings.models_dir / "ctd"
    return d.is_dir() and any(d.glob("*.onnx"))


def test_ctd_runs_without_crashing():
    if not _model_available():
        return "SKIP: CTD onnx weights not present"
    from app.registry import registry

    detector = registry.get("detector", "ctd")
    img = Image.new("RGB", (800, 1200), (255, 255, 255))
    regions = detector.detect(img, {})
    assert isinstance(regions, list)


def test_ctd_detects_text_on_fixture():
    if not (_model_available() and os.environ.get("SCANLATION_CTD_FIXTURE")):
        return "SKIP: needs weights + SCANLATION_CTD_FIXTURE manga page"
    from app.registry import registry

    detector = registry.get("detector", "ctd")
    img = Image.open(os.environ["SCANLATION_CTD_FIXTURE"]).convert("RGB")
    regions = detector.detect(img, {})
    assert len(regions) > 0
    for r in regions:
        assert r.polygon.shape == (4, 2)


TESTS = [test_ctd_runs_without_crashing, test_ctd_detects_text_on_fixture]

if __name__ == "__main__":
    import sys

    from tests.helpers import run

    sys.exit(run(TESTS, "test_ctd (slow)"))
