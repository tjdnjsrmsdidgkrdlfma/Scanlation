"""The engine contract — the single seam every plugin plugs into.

The key design departure from the old (Crivella) stack: a Region carries
*rotated* geometry (a 4-point polygon + angle + optional mask), not just an
axis-aligned box. This lets the pipeline deskew tilted text (SFX, vertical
JP) before recognition. Axis-aligned detectors degrade gracefully via
``Region.from_bbox`` (a right-angled quad, angle 0).

Only ``bbox`` (= [x_min, y_min, x_max, y_max]) is serialized to the wire, which
the browser extension reads as [l, b, r, t]. polygon/angle/mask stay
server-internal (deskew, future inpaint).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from PIL import Image


@dataclass
class Region:
    polygon: np.ndarray                     # (4, 2) float32, image px, possibly rotated
    bbox: tuple[int, int, int, int]         # axis-aligned (x_min, y_min, x_max, y_max), derived
    angle: float = 0.0                      # signed deskew angle (deg)
    vertical: bool = False                  # Japanese vertical writing
    mask: Optional[np.ndarray] = None       # optional per-region segmentation mask
    score: float = 1.0
    order: int = 0                          # reading order (assigned by pipeline)
    label: str = ""                         # detector class (e.g. text_bubble/text_free); "" if unclassified. server-internal, not on the wire

    @staticmethod
    def _bbox_from_poly(poly: np.ndarray) -> tuple[int, int, int, int]:
        xs, ys = poly[:, 0], poly[:, 1]
        return (
            int(np.floor(float(xs.min()))),
            int(np.floor(float(ys.min()))),
            int(np.ceil(float(xs.max()))),
            int(np.ceil(float(ys.max()))),
        )

    @classmethod
    def from_quad(cls, quad, **kw) -> "Region":
        """Build from any 4-point quad (rotated or not)."""
        poly = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        return cls(polygon=poly, bbox=cls._bbox_from_poly(poly), **kw)

    @classmethod
    def from_bbox(cls, x_min, y_min, x_max, y_max, **kw) -> "Region":
        """Build from an axis-aligned box (degenerate right-angled quad, angle 0).

        Vertices in clockwise order from top-left, matching the (W,H) target
        deskew uses.
        """
        quad = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        return cls.from_quad(quad, **kw)

    def wire_box(self) -> list[int]:
        """The 4-list the extension consumes: [x_min, y_min, x_max, y_max]."""
        return [int(v) for v in self.bbox]


# --- Lifecycle / metadata mixin shared by every plugin ---------------------
class EngineBase:
    name: str = "base"
    display_name: str = "Base engine"
    homepage: Optional[str] = None
    warning: Optional[str] = None
    description: str = ""
    # {opt_name: {"type": <python type>, "default": <literal>, "description": str}}
    OPTION_SCHEMA: dict = {}
    SUPPORTED_SRC: list[str] = []           # iso1 codes, [] = any
    SUPPORTED_DST: list[str] = []

    @property
    def _log(self) -> logging.Logger:
        """Per-engine logger, namespaced scanlation.<name> — shared by every plugin."""
        return logging.getLogger(f"scanlation.{self.name}")

    def is_installed(self) -> bool:
        """Are this engine's local resources (weights) present? Engines with no
        downloadable assets (dummy, external services) stay True."""
        return True

    def list_models(self) -> list[str]:
        """Model tags/names selectable from this engine's backend, for the admin
        model picker (e.g. an LLM translator listing its server's pulled models).
        [] = not applicable or the backend couldn't be reached; never raises."""
        return []

    def install(self) -> None:
        """Fetch/prepare local resources — the explicit 'install' action (one-click
        from the popup / manage_plugins, or tools/install.py). No-op by default;
        load() never downloads implicitly."""

    def load(self) -> None:
        """Acquire heavy resources (VRAM/model) from already-installed assets.
        Called lazily on first use; raises if not installed."""

    def unload(self) -> None:
        """Release resources."""

    def resolve_options(self, options: Optional[dict]) -> dict:
        """Fill any option the caller left unset (missing or blank ""/None) with
        this engine's OPTION_SCHEMA default, and coerce every schema option to its
        declared type. OPTION_SCHEMA is thus the single source of defaults: the
        admin-shown default always matches what the engine actually runs, and
        engines stay self-contained (they work when called with {})."""
        out = dict(options or {})
        for key, spec in type(self).OPTION_SCHEMA.items():
            if out.get(key, "") in ("", None) and "default" in spec:
                out[key] = spec["default"]
            t = spec.get("type")
            if t in (int, float, bool, str) and key in out:
                try:
                    out[key] = t(out[key])
                except (TypeError, ValueError):
                    pass
        return out


@runtime_checkable
class Detector(Protocol):
    def detect(self, image: Image.Image, options: dict) -> list[Region]: ...


@runtime_checkable
class Recognizer(Protocol):
    # ``crop`` is already deskewed upright by the pipeline.
    def recognize(self, crop: Image.Image, region: Region, options: dict) -> str: ...


@runtime_checkable
class Translator(Protocol):
    def translate(self, text: str, src: str, dst: str, options: dict) -> str: ...


@runtime_checkable
class BatchTranslator(Translator, Protocol):
    """A translator that can render a whole image's texts in ONE model call.

    Optional: the pipeline tests for this protocol and falls back to a per-text
    ``translate`` loop when a translator doesn't implement it. ``translate_batch``
    returns one translation per input, aligned to the input order, and is expected
    to fall back internally rather than raise (see ``HttpTranslatorBase``).
    """

    def translate_batch(
        self, texts: list[str], src: str, dst: str, options: dict
    ) -> list[str]: ...
