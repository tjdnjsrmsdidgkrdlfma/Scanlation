"""Dummy engines — the permanent test doubles.

These ship in the package (not throwaway): they make the whole skeleton — wire
protocol, pipeline, cache, reading order, routes — testable with zero models,
deterministically and fast. The dummy detector deliberately emits one rotated
quad so the deskew path is exercised end-to-end.
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
