"""Contract/Region invariants + plugin protocol conformance + the SDK's
device policy and local-model lifecycle base."""
from __future__ import annotations

import numpy as np

from scanlation_sdk.context import context
from scanlation_sdk.contracts import Detector, Recognizer, Region, Translator
from scanlation_sdk.device import pick_device
from scanlation_sdk.local_engine import LocalModelEngineBase
from tests.fake_engines import DummyDetector, DummyRecognizer, DummyTranslator


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


# --- SDK device policy (shared by every local-model engine) ----------------
def test_pick_device_cpu_is_pinned():
    """device 'cpu' -> always CPU, regardless of GPU presence (case-insensitive)."""
    saved = context.device
    try:
        context.device = "cpu"
        assert pick_device() == "cpu"
        context.device = "CPU"
        assert pick_device() == "cpu"
    finally:
        context.device = saved


def test_pick_device_gpu_uses_cuda_when_available():
    """device != 'cpu' -> cuda if torch reports it, else a safe CPU fallback.
    Computed against the same torch check so it's deterministic on any host."""
    saved = context.device
    try:
        context.device = "cuda"
        expected = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                expected = "cuda"
        except Exception:  # noqa: BLE001 - no torch -> stays cpu
            pass
        assert pick_device() == expected
    finally:
        context.device = saved


# --- LocalModelEngineBase lifecycle -----------------------------------------
def test_local_engine_lifecycle():
    """The base enforces install/load guards and the never-download-in-load rule."""

    class Fake(LocalModelEngineBase):
        name = "fake"
        INSTALL_HINT = "Run the fake installer."

        def __init__(self):
            self.installed = False
            self.downloads = 0
            self.loads = 0
            self.model = None

        def is_installed(self):
            return self.installed

        def _download(self):
            self.downloads += 1
            self.installed = True

        def _load(self, device):
            self.loads += 1
            self.model = device  # any truthy resource

        def _unload(self):
            self.model = None

    e = Fake()
    # load before install -> refuses with the hint (never downloads implicitly)
    try:
        e.load()
        raise AssertionError("load() must raise when not installed")
    except RuntimeError as exc:
        assert "fake weights not installed" in str(exc)
        assert "Run the fake installer." in str(exc)
    assert e.downloads == 0

    e.install()
    assert e.downloads == 1
    e.install()  # idempotent: already installed -> no re-download
    assert e.downloads == 1

    e.load()
    e.load()  # loaded guard: second call is a no-op
    assert e.loads == 1 and e.model is not None

    e.unload()
    assert e.model is None
    e.load()  # reload after unload works
    assert e.loads == 2


TESTS = [
    test_region_from_bbox_shape_and_wire,
    test_region_from_quad_bbox_is_enclosing,
    test_dummy_engines_satisfy_protocols,
    test_dummy_detector_emits_rotated_region,
    test_pick_device_cpu_is_pinned,
    test_pick_device_gpu_uses_cuda_when_available,
    test_local_engine_lifecycle,
]

if __name__ == "__main__":
    import sys

    from tests.helpers import run

    sys.exit(run(TESTS, "test_contracts"))
