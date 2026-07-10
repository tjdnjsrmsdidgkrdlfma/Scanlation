"""The core pipeline: detect -> reading order -> deskew -> recognize -> translate.

Runs synchronously (the route layer runs it in a threadpool under the GPU lock).
Engine *instances* and per-engine option dicts are passed in, so the pipeline is
fully decoupled from selection/registry — which is what lets tests drive it with
dummy engines.
"""
from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from PIL import Image

from scanlation_sdk.context import LANG_RTL
from scanlation_sdk.contracts import BatchTranslator, Detector, Recognizer, Region, Translator
from .geometry import deskew_crop

logger = logging.getLogger("scanlation.pipeline")


class ResultItem(TypedDict):
    """One translated region, exactly as the extension reads it (see
    ``extension/src/content.js``). Built in ``translate_regions`` and nowhere else —
    the routes pass the list straight through, so this is the wire item's only
    definition on the server side."""
    bounds: list[int]   # [x_min, y_min, x_max, y_max]; the client reads it as [l, b, r, t]
    source: str         # what the recognizer read
    destination: str    # what the translator produced


def assign_reading_order(regions: list[Region], *, rtl: bool) -> list[Region]:
    """Reading order: top-to-bottom rows, and within a row right-to-left when the
    source language's comics read that way (``rtl``) — Japanese manga — else
    left-to-right.

    Rows are banded by the median region height so slightly misaligned bubbles
    still group into the same row.

    This order is what the translator receives: a page's bubbles go to the LLM as
    one sequence, so their sequence is the context each translation is read in.
    """
    if not regions:
        return regions
    heights = sorted(r.bbox[3] - r.bbox[1] for r in regions)
    band = max(1, heights[len(heights) // 2])
    across = -1 if rtl else 1

    def key(r: Region):
        x0, y0, x1, y1 = r.bbox
        return (int(((y0 + y1) / 2) // band), across * ((x0 + x1) / 2))

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
    opt_detect: dict[str, Any],
    opt_recognize: dict[str, Any],
) -> tuple[list[tuple[str, Region]], dict[str, float]]:
    """Detect regions, order them, deskew+recognize each. Returns ``(pairs, timing)``:
    non-empty (text, region) pairs in reading order, plus ``{detect_ms, recognize_ms}``
    (the two halves, also logged — the caller surfaces them in the run response). This
    is the GPU/model half of the pipeline — the route runs it under the GPU lock.
    assign_reading_order is called exactly once here (it assigns region.order)."""
    t0 = time.perf_counter()
    regions = assign_reading_order(detector.detect(img, opt_detect), rtl=(src in LANG_RTL))
    t_det = time.perf_counter()
    out: list[tuple[str, Region]] = []
    for region in regions:
        crop = deskew_crop(img, region)
        t_crop = time.perf_counter()
        text = recognizer.recognize(crop, region, opt_recognize).strip()
        # One line per region (verbose/DEBUG): what was detected, where, its class,
        # per-crop recognize time, and what the recognizer read — logged even when empty
        # ("detected but recognized nothing" is itself a signal). See /admin 동작.
        logger.debug(
            "  #%d %s bbox=%s%s score=%.2f %.0fms -> %r",
            region.order, getattr(region, "label", "") or "?", region.bbox,
            " vert" if region.vertical else "", region.score,
            (time.perf_counter() - t_crop) * 1000, text,
        )
        if text:
            out.append((text, region))
    t_rec = time.perf_counter()
    logger.info(
        "detect %s: %d regions %.0fms | recognize %s: %d texts %.0fms",
        getattr(detector, "name", "?"), len(regions), (t_det - t0) * 1000,
        getattr(recognizer, "name", "?"), len(out), (t_rec - t_det) * 1000,
    )
    return out, {"detect_ms": round((t_det - t0) * 1000, 1),
                 "recognize_ms": round((t_rec - t_det) * 1000, 1)}


def _translate_all(
    texts: list[str], src: str, dst: str, translator: Translator, options: dict
) -> list[str]:
    """Translate a whole image's texts. Uses the translator's batch path when it
    has one (the LLM engines) and a per-text loop otherwise (the test dummy)."""
    if isinstance(translator, BatchTranslator):
        return translator.translate_batch(texts, src, dst, options)
    return [translator.translate(t, src, dst, options) for t in texts]


def translate_regions(
    recognized: list[tuple[str, Region]],
    *,
    translator: Translator,
    src: str,
    dst: str,
    opt_translate: dict[str, Any],
) -> list[ResultItem]:
    """Translate recognized (text, region) pairs -> the wire result list. This is
    the LLM half of the pipeline — the route runs it OUTSIDE the GPU lock."""
    if not recognized:
        logger.info("translate: 0 texts (nothing recognized)")
        return []
    texts = [text for text, _ in recognized]
    t0 = time.perf_counter()
    translations = _translate_all(texts, src, dst, translator, opt_translate)
    logger.info(
        "translate %s: %d texts %.0fms",
        getattr(translator, "name", "?"), len(texts), (time.perf_counter() - t0) * 1000,
    )
    for i, (src_text, dst_text) in enumerate(zip(texts, translations)):
        logger.debug("  t%d %r -> %r", i, src_text, dst_text)  # verbose: source -> translation
    return [
        ResultItem(bounds=region.wire_box(), source=text, destination=translation)
        for (text, region), translation in zip(recognized, translations)
    ]


def run_pipeline(
    img: Image.Image,
    *,
    detector: Detector,
    recognizer: Recognizer,
    translator: Translator,
    src: str,
    dst: str,
    opt_detect: dict[str, Any],
    opt_recognize: dict[str, Any],
    opt_translate: dict[str, Any],
) -> list[ResultItem]:
    """Detect+recognize then translate — the composed reference path (tests, and
    any single-call use). The route splits these two halves across the GPU lock."""
    recognized, _timing = detect_and_recognize(
        img, detector=detector, recognizer=recognizer, src=src, opt_detect=opt_detect, opt_recognize=opt_recognize
    )
    return translate_regions(recognized, translator=translator, src=src, dst=dst, opt_translate=opt_translate)
