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
from scanlation_sdk.contracts import Detector, Recognizer, Region, Translator
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
    """Translate one text and record the machine result in the TM (for /get_trans/)."""
    tsl = translator.translate(text, src, dst, options)
    model = getattr(translator, "name", "machine")
    cache.put_translation(text, src, dst, model, tsl)
    return tsl


def detect_and_recognize(
    img: Image.Image,
    *,
    detector: Detector,
    recognizer: Recognizer,
    src: str,
    opt_box: dict[str, Any],
    opt_ocr: dict[str, Any],
) -> list[tuple[str, Region]]:
    """Detect regions, order them, deskew+recognize each. Returns non-empty
    (text, region) pairs in reading order. This is the GPU/model half of the
    pipeline — the route runs it under the GPU lock. assign_reading_order is
    called exactly once here (it assigns region.order)."""
    regions = assign_reading_order(detector.detect(img, opt_box), vertical_hint=(src == "ja"))
    out: list[tuple[str, Region]] = []
    for region in regions:
        crop = deskew_crop(img, region)
        text = recognizer.recognize(crop, region, opt_ocr).strip()
        if text:
            out.append((text, region))
    return out


def _translate_all(
    texts: list[str], src: str, dst: str, translator: Translator, options: dict
) -> list[str]:
    """Translate a whole image's texts. Uses the translator's batch path when it
    has one (the LLM engines) and a per-text loop otherwise (the test dummy).
    Records each source->translation in the TM, one row per text keyed by the
    engine name — same as the single-text translate_text."""
    if hasattr(translator, "translate_batch"):
        tsls = translator.translate_batch(texts, src, dst, options)
    else:
        tsls = [translator.translate(t, src, dst, options) for t in texts]
    model = getattr(translator, "name", "machine")
    for text, tsl in zip(texts, tsls):
        cache.put_translation(text, src, dst, model, tsl)
    return tsls


def translate_regions(
    recognized: list[tuple[str, Region]],
    *,
    translator: Translator,
    src: str,
    dst: str,
    opt_tsl: dict[str, Any],
) -> list[dict]:
    """Translate recognized (text, region) pairs -> the wire result list. This is
    the LLM half of the pipeline — the route runs it OUTSIDE the GPU lock."""
    if not recognized:
        return []
    texts = [text for text, _ in recognized]
    tsls = _translate_all(texts, src, dst, translator, opt_tsl)
    return [
        {"ocr": text, "tsl": tsl, "box": region.wire_box()}
        for (text, region), tsl in zip(recognized, tsls)
    ]


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
    """Detect+recognize then translate — the composed reference path (tests, and
    any single-call use). The route splits these two halves across the GPU lock."""
    recognized = detect_and_recognize(
        img, detector=detector, recognizer=recognizer, src=src, opt_box=opt_box, opt_ocr=opt_ocr
    )
    return translate_regions(recognized, translator=translator, src=src, dst=dst, opt_tsl=opt_tsl)
