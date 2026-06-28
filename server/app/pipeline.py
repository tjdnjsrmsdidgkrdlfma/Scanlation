"""The core pipeline: detect -> reading order -> deskew -> recognize -> translate.

Runs synchronously (the route layer runs it in a threadpool under the GPU lock).
Engine *instances* and per-engine option dicts are passed in, so the pipeline is
fully decoupled from selection/registry — which is what lets tests drive it with
dummy engines.
"""
from __future__ import annotations

from typing import Any

from PIL import Image

from .cache import cache
from .config import settings
from .contracts import Detector, Recognizer, Region, Translator
from .geometry import deskew_crop


def assign_reading_order(regions: list[Region], vertical_hint: bool = False) -> list[Region]:
    """Manga reading order: top-to-bottom rows, right-to-left within a row.

    Rows are banded by the median region height so slightly misaligned bubbles
    still group into the same row.
    """
    if not regions:
        return regions
    heights = sorted(r.bbox[3] - r.bbox[1] for r in regions)
    band = max(1, heights[len(heights) // 2])

    def key(r: Region):
        x0, y0, x1, y1 = r.bbox
        return (int(((y0 + y1) / 2) // band), -((x0 + x1) / 2))

    ordered = sorted(regions, key=key)
    for i, r in enumerate(ordered):
        r.order = i
    return ordered


def translate_text(
    text: str, src: str, dst: str, translator: Translator, options: dict
) -> str:
    """Translate one text, honoring manual TM overrides and caching machine output."""
    best = cache.best_translation(text, src, dst)
    if best is not None and best["model"] == "manual":
        return best["text"]
    tsl = translator.translate(text, src, dst, options)
    model = getattr(translator, "name", "machine")
    cache.put_translation(text, src, dst, model, tsl)
    return tsl


def run_pipeline(
    img: Image.Image,
    *,
    detector: Detector,
    recognizer: Recognizer,
    translator: Translator,
    src: str,
    dst: str,
    opt_box: dict[str, Any],
    opt_ocr: dict[str, Any],
    opt_tsl: dict[str, Any],
) -> list[dict]:
    """Return the wire result list: [{"ocr", "tsl", "box"=[x0,y0,x1,y1]}]."""
    regions = detector.detect(img, opt_box)
    regions = assign_reading_order(regions, vertical_hint=(src == "ja"))

    out: list[dict] = []
    for region in regions:
        crop = deskew_crop(img, region)
        text = recognizer.recognize(crop, region, opt_ocr).strip()
        if not text:
            continue
        tsl = translate_text(text, src, dst, translator, opt_tsl)
        out.append({"ocr": text, "tsl": tsl, "box": region.wire_box()})
    return out
