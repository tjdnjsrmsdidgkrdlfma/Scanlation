"""Test-only fake engines.

The product ships no placeholder engine — a fresh install has no engines until
you install a real one, and running without one is a 400 error.
These fakes exist ONLY for the test suite: they exercise the whole skeleton —
wire protocol, pipeline, cache, reading order, routes — with zero models,
deterministically and fast. The fake detector deliberately emits one rotated
quad so the deskew path is covered. ``install_fakes()`` registers them into the
live registry + selection so the route tests have a working pipeline.

They keep ``name = "dummy"`` so the existing tests read naturally.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image

from scanlation_sdk.contracts import EngineBase, Region


def _rotated_quad(cx: float, cy: float, w: float, h: float, angle_deg: float) -> list[list[float]]:
    """Corners (TL, TR, BR, BL) of a w*h box centered at (cx,cy), rotated angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    base = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return [[cx + x * ca - y * sa, cy + x * sa + y * ca] for x, y in base]


class DummyDetector(EngineBase):
    name = "dummy"
    display_name = "Dummy detector"
    description = "Deterministic hardcoded regions (one rotated) for protocol/pipeline tests."
    OPTION_SCHEMA = {
        "num_boxes": {"type": int, "default": 2, "description": "How many fake regions to emit (1-2)."},
    }

    def detect(self, image: Image.Image, options: dict[str, Any]) -> list[Region]:
        w, h = image.size
        n = int(options.get("num_boxes", 2))
        regions = [
            # Top-right axis-aligned box -> reading order 0 (manga R->L).
            Region.from_bbox(0.55 * w, 0.10 * h, 0.85 * w, 0.22 * h, score=0.99),
            # Top-left box rotated 12deg -> exercises deskew warp.
            Region.from_quad(
                _rotated_quad(0.27 * w, 0.16 * h, 0.26 * w, 0.10 * h, 12.0),
                angle=12.0, score=0.95,
            ),
        ]
        return regions[: max(1, min(n, len(regions)))]


class DummyRecognizer(EngineBase):
    name = "dummy"
    display_name = "Dummy recognizer"
    description = "Returns REGION-<order> so pipeline output is deterministic."

    def recognize(self, crop: Image.Image, region: Region, options: dict[str, Any]) -> str:
        return f"REGION-{region.order}"


class DummyTranslator(EngineBase):
    name = "dummy"
    display_name = "Dummy translator"
    description = "Echoes [src->dst] text without any model."
    SUPPORTED_SRC: list[str] = []  # any
    SUPPORTED_DST: list[str] = []

    def translate(self, text: str, src: str, dst: str, options: dict[str, Any]) -> str:
        return f"[{src}->{dst}] {text}"


class _FakePool:
    """In-process stand-in for a worker pool. Both halves now ALWAYS go through one
    (orchestrator._detect_sync / _recognize_sync), and the spawn workers re-discover
    their engine by ENTRY POINT — but these fakes are registered straight into the
    registry, not pip-installed, so a real worker can't find "dummy". This runs the
    registered fake in-process instead (no subprocess). Subclasses supply ``run``."""
    def ensure(self, name, device, workers=1): pass
    def invalidate(self, name=None): pass
    def shutdown(self): pass
    def idle_seconds(self, now): return None


class _FakeDetectPool(_FakePool):
    """items = [(image, opt)] -> [list[Region]] — same contract as EnginePool.run for
    the detect task (one page per run, raw regions; the caller orders them)."""
    def run(self, items):
        det = DummyDetector()
        return [det.detect(img, opt) for img, opt in items]


class _FakeRecognizePool(_FakePool):
    """items = [(crop, opt), ...] -> [(text, ms), ...]. Injects region.order = i (the
    crop's index == its region.order, since detect returns regions in reading order),
    so the golden "REGION-<order>" output is reproduced exactly."""
    def run(self, items):
        rec = DummyRecognizer()
        out = []
        for i, (crop, opt) in enumerate(items):
            region = Region.from_bbox(0, 0, crop.width, crop.height)
            region.order = i
            out.append((rec.recognize(crop, region, opt).strip(), 1.0))
        return out


def install_fakes() -> None:
    """Register the fakes into the live registry + select them, so route tests
    have a working detector/recognizer/translator (the product ships none)."""
    from app.registry import registry
    from app.state import state

    registry.all_classes()["detector"]["dummy"] = DummyDetector
    registry.all_classes()["recognizer"]["dummy"] = DummyRecognizer
    registry.all_classes()["translator"]["dummy"] = DummyTranslator
    state.selection.detector = "dummy"
    state.selection.recognizer = "dummy"
    state.selection.translator = "dummy"

    # Both halves now ALWAYS run through a worker pool. Rebind ONLY orchestrator's
    # names to the in-process fakes, so route tests exercise the always-pool paths
    # without spawning workers (and without touching the real singletons, which
    # test_idle_unload/test_engine_pool drive directly). Idempotent so client()'s
    # cache can call install_fakes again harmlessly.
    import app.orchestrator as _orch
    if not isinstance(_orch.detect_pool, _FakeDetectPool):
        _orch.detect_pool = _FakeDetectPool()
    if not isinstance(_orch.recognize_pool, _FakeRecognizePool):
        _orch.recognize_pool = _FakeRecognizePool()
