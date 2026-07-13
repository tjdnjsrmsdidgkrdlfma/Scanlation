"""The core pipeline: detect -> reading order -> deskew -> recognize -> translate.

Runs synchronously (the route layer runs it in a threadpool under the GPU lock).
Engine *instances* and per-engine option dicts are passed in, so the pipeline is
fully decoupled from selection/registry — which is what lets tests drive it with
dummy engines.
"""
from __future__ import annotations

import logging
import time
from contextlib import nullcontext
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
    recognizer: Recognizer | None,
    src: str,
    opt_detect: dict[str, Any],
    opt_recognize: dict[str, Any],
    pool: Any = None,
    rec_name: str | None = None,
    detect_lock: Any = None,
) -> tuple[list[tuple[str, Region]], dict[str, float]]:
    """Detect regions, order them, deskew+recognize each. Returns ``(pairs, timing)``:
    non-empty (text, region) pairs in reading order, plus ``{detect_ms, recognize_ms}``
    (the two halves, also logged — the caller surfaces them in the run response). This
    is the GPU/model half of the pipeline — the route runs it under the GPU lock.
    assign_reading_order is called exactly once here (it assigns region.order).

    Recognition takes one of two paths — the same optional-capability seam as
    ``_translate_all``: with a ``pool`` (a RecognizePool) a page's crops fan out
    across worker processes (each B=1) and ``recognizer`` is unused (the workers own
    it — pass its name as ``rec_name`` for the log); without one, the in-process
    per-crop loop runs on ``recognizer`` directly (the default; tests and CPU
    engines). ``detect_lock`` (optional) serializes ``detector.detect`` across
    concurrent readers — the detector is a shared in-process torch model; None = no
    serialization (single-reader path, tests)."""
    t0 = time.perf_counter()
    with detect_lock or nullcontext():  # shared torch detector: one forward at a time
        regions = assign_reading_order(detector.detect(img, opt_detect), rtl=(src in LANG_RTL))
    t_det = time.perf_counter()
    out, details = (_recognize_via_pool(img, regions, opt_recognize, pool) if pool is not None
                    else _recognize_per_crop(img, regions, recognizer, opt_recognize))
    t_rec = time.perf_counter()
    logger.info(
        "detect %s: %d regions %.0fms | recognize %s: %d texts %.0fms",
        getattr(detector, "name", "?"), len(regions), (t_det - t0) * 1000,
        rec_name or getattr(recognizer, "name", "?"), len(out), (t_rec - t_det) * 1000,
    )
    # region_details/raw_regions ride the timing dict (signature stays (out, timing));
    # the orchestrator pops them for the stats DB. raw_regions = detected boxes (incl.
    # ones recognized empty); len(out) = the non-empty ones.
    return out, {"detect_ms": round((t_det - t0) * 1000, 1),
                 "recognize_ms": round((t_rec - t_det) * 1000, 1),
                 "raw_regions": len(regions), "region_details": details}


def _region_detail(region: Region, text: str, ms: float) -> dict:
    """One crop's stat row: box w/h (area·aspect·vertical all derive from these),
    detection score, recognized-text length, and this crop's recognize time. dest_len
    is filled in later (post-translate), so it's not here."""
    x0, y0, x1, y1 = region.bbox
    return {"crop_w": x1 - x0, "crop_h": y1 - y0, "score": region.score,
            "source_len": len(text), "recognize_ms": round(ms, 1)}


def _recognize_per_crop(
    img: Image.Image, regions: list[Region], recognizer: Recognizer, opt_recognize: dict
) -> tuple[list[tuple[str, Region]], list[dict]]:
    """In-process path: deskew + recognize one crop at a time (the default, and the
    fallback for CPU/test engines). Returns (non-empty (text, region) pairs, per-crop
    stat details for every detected region)."""
    out: list[tuple[str, Region]] = []
    details: list[dict] = []
    for region in regions:
        crop = deskew_crop(img, region)
        t_crop = time.perf_counter()
        text = recognizer.recognize(crop, region, opt_recognize).strip()
        ms = (time.perf_counter() - t_crop) * 1000
        details.append(_region_detail(region, text, ms))
        # One line per region (verbose/DEBUG): what was detected, where, its class,
        # per-crop recognize time, and what the recognizer read — logged even when empty
        # ("detected but recognized nothing" is itself a signal). See /admin 동작.
        logger.debug(
            "  #%d %s bbox=%s%s score=%.2f %.0fms -> %r",
            region.order, getattr(region, "label", "") or "?", region.bbox,
            " vert" if region.vertical else "", region.score, ms, text,
        )
        if text:
            out.append((text, region))
    return out, details


def _recognize_via_pool(
    img: Image.Image, regions: list[Region], opt_recognize: dict, pool: Any
) -> tuple[list[tuple[str, Region]], list[dict]]:
    """Worker-pool path: deskew every crop in-process (CPU), then recognize them all
    across the pool's worker processes (each B=1, returning (text, ms)). Order is
    preserved by pool.run, so results zip back onto ``regions`` in reading order.
    Returns (non-empty pairs, per-crop stat details for every detected region)."""
    crops = [deskew_crop(img, region) for region in regions]
    results = pool.run([(crop, opt_recognize) for crop in crops])
    out: list[tuple[str, Region]] = []
    details: list[dict] = []
    for (text, ms), region in zip(results, regions):
        details.append(_region_detail(region, text, ms))
        logger.debug("  #%d %s bbox=%s%s %.0fms -> %r", region.order,
                     getattr(region, "label", "") or "?", region.bbox,
                     " vert" if region.vertical else "", ms, text)
        if text:
            out.append((text, region))
    return out, details


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


def recognized_to_result(recognized: list[tuple[str, Region]]) -> list[ResultItem]:
    """Wire result for a recognize-only run (skip_translate): recognized text as
    ``source`` with an empty ``destination`` — the same shape as translate_regions
    minus the LLM call. Lets a benchmark measure detect+recognize without a translator
    running (e.g. when a GPU VLM recognizer and the LLM can't share VRAM)."""
    return [
        ResultItem(bounds=region.wire_box(), source=text, destination="")
        for text, region in recognized
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
