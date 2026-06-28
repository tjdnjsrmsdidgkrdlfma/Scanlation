"""CTD smoke test (slow). Skipped unless ONNX weights are present.

Set SCANLATION_CTD_MODEL or drop an .onnx into <models>/ctd/. Provide a real
manga page via SCANLATION_CTD_FIXTURE to assert region count > 0.
"""
from __future__ import annotations

import os

import pytest
from PIL import Image

pytestmark = pytest.mark.slow


def _model_available() -> bool:
    if os.environ.get("SCANLATION_CTD_MODEL"):
        return True
    from app.config import settings

    d = settings.models_dir / "ctd"
    return d.is_dir() and any(d.glob("*.onnx"))


@pytest.mark.skipif(not _model_available(), reason="CTD onnx weights not present")
def test_ctd_runs_without_crashing():
    from app.registry import registry

    detector = registry.get("detector", "ctd")
    img = Image.new("RGB", (800, 1200), (255, 255, 255))
    regions = detector.detect(img, {})
    assert isinstance(regions, list)


@pytest.mark.skipif(
    not (_model_available() and os.environ.get("SCANLATION_CTD_FIXTURE")),
    reason="needs weights + SCANLATION_CTD_FIXTURE manga page",
)
def test_ctd_detects_text_on_fixture():
    from app.registry import registry

    detector = registry.get("detector", "ctd")
    img = Image.open(os.environ["SCANLATION_CTD_FIXTURE"]).convert("RGB")
    regions = detector.detect(img, {})
    assert len(regions) > 0
    for r in regions:
        assert r.polygon.shape == (4, 2)
