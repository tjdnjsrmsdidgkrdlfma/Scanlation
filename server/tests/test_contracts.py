"""Contract/Region invariants + plugin protocol conformance."""
from __future__ import annotations

import numpy as np

from app.contracts import Detector, Recognizer, Region, Translator
from plugins.dummy.plugin import DummyDetector, DummyRecognizer, DummyTranslator


def test_region_from_bbox_shape_and_wire():
    r = Region.from_bbox(10, 20, 110, 70)
    assert r.polygon.shape == (4, 2)
    assert r.bbox == (10, 20, 110, 70)
    assert r.wire_box() == [10, 20, 110, 70]  # == client [l, b, r, t]
    assert r.angle == 0.0


def test_region_from_quad_bbox_is_enclosing():
    quad = [[10, 0], [30, 10], [20, 30], [0, 20]]  # diamond
    r = Region.from_quad(quad, angle=15.0)
    assert r.bbox == (0, 0, 30, 30)
    assert r.angle == 15.0


def test_dummy_engines_satisfy_protocols():
    assert isinstance(DummyDetector(), Detector)
    assert isinstance(DummyRecognizer(), Recognizer)
    assert isinstance(DummyTranslator(), Translator)


def test_dummy_detector_emits_rotated_region():
    img_like = type("I", (), {"size": (400, 300)})()
    regions = DummyDetector().detect(img_like, {})
    assert len(regions) == 2
    angles = [r.angle for r in regions]
    assert any(abs(a) > 1.0 for a in angles)  # at least one rotated quad
    for r in regions:
        assert isinstance(r.polygon, np.ndarray) and r.polygon.shape == (4, 2)
