"""The core pipeline: detect -> reading order -> deskew -> recognize -> translate.

Runs synchronously (the route layer runs it in a threadpool under the GPU lock).
Engine *instances* and per-engine option dicts are passed in, so the pipeline is
fully decoupled from selection/registry — which is what lets tests drive it with
dummy engines.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from PIL import Image

from scanlation_sdk.contracts import Detector, Recognizer, Region, Translator
from .geometry import deskew_crop

logger = logging.getLogger("scanlation.pipeline")


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
    t0 = time.perf_counter()
    regions = assign_reading_order(detector.detect(img, opt_box), vertical_hint=(src == "ja"))
    t_det = time.perf_counter()
    out: list[tuple[str, Region]] = []
    for region in regions:
        crop = deskew_crop(img, region)
        text = recognizer.recognize(crop, region, opt_ocr).strip()
        if text:
            out.append((text, region))
    logger.info(
        "detect %s: %d regions %.0fms | recognize %s: %d texts %.0fms",
        getattr(detector, "name", "?"), len(regions), (t_det - t0) * 1000,
        getattr(recognizer, "name", "?"), len(out), (time.perf_counter() - t_det) * 1000,
    )
    return out


def _translate_all(
    texts: list[str], src: str, dst: str, translator: Translator, options: dict
) -> list[str]:
    """Translate a whole image's texts. Uses the translator's batch path when it
    has one (the LLM engines) and a per-text loop otherwise (the test dummy)."""
    if hasattr(translator, "translate_batch"):
        return translator.translate_batch(texts, src, dst, options)
    return [translator.translate(t, src, dst, options) for t in texts]


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
        logger.info("translate: 0 texts (nothing recognized)")
        return []
    texts = [text for text, _ in recognized]
    t0 = time.perf_counter()
    tsls = _translate_all(texts, src, dst, translator, opt_tsl)
    logger.info(
        "translate %s: %d texts %.0fms",
        getattr(translator, "name", "?"), len(texts), (time.perf_counter() - t0) * 1000,
    )
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
