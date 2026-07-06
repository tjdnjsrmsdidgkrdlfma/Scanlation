"""Pipeline orchestration behind /run_pipeline/ and /run_lookup/.

Selection -> plan, cache identity, GPU-lock sequencing, and in-flight dedup live
here, HTTP-free: the route validates the request and maps errors to status
codes; this module just runs the work. A ``RunPlan`` bundles the resolved engine
selection + merged options + the cache-identity hash so the route and the
compute path share one object instead of re-resolving.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from dataclasses import dataclass

from PIL import Image
from starlette.concurrency import run_in_threadpool

from .cache import cache, opt_hash
from .pipeline import detect_and_recognize, translate_regions
from .registry import registry
from .state import state

logger = logging.getLogger("scanlation.run")


class BadImageError(ValueError):
    """Undecodable request image (the route maps this to a 400)."""


@dataclass(frozen=True)
class RunPlan:
    """One request's execution plan + cache identity (from the current selection
    merged with the request's option overrides)."""
    detector: str
    recognizer: str
    translator: str
    src: str
    dst: str
    engines: str            # "det+rec+tr"
    opt_detect: dict
    opt_recognize: dict
    opt_translate: dict
    oh: str                 # opt_hash(opt_detect, opt_recognize, opt_translate) — cache key part

    @property
    def cache_key(self) -> tuple:
        """The tuple that identifies this run in the cache + the in-flight map."""
        return (self.src, self.dst, self.engines, self.oh)


def make_plan(request_options: dict | None) -> RunPlan:
    """Resolve the current selection + merge this request's option overrides into
    a RunPlan (request options win over the persisted per-engine overrides)."""
    sel = state.selection
    det, rec, tr = sel.detector, sel.recognizer, sel.translator
    opt_detect = state.options_for(det, request_options)
    opt_recognize = state.options_for(rec, request_options)
    opt_translate = state.translator_options(tr, request_options)
    return RunPlan(
        detector=det, recognizer=rec, translator=tr,
        src=sel.lang_src, dst=sel.lang_dst,
        engines=f"{det}+{rec}+{tr}",
        opt_detect=opt_detect, opt_recognize=opt_recognize, opt_translate=opt_translate,
        oh=opt_hash(opt_detect, opt_recognize, opt_translate),
    )


def cached_result(plan: RunPlan, md5: str) -> list | None:
    """The cached page result for this plan+image, or None on a miss."""
    return cache.get_run(md5, *plan.cache_key)


def _read_sync(img, plan: RunPlan):
    """Read the image = detect + recognize (the GPU/model half). Resolves all three
    engines (loads weights on first use); returns (recognized pairs, translator).

    Runs in the threadpool UNDER the GPU lock: registry.get is not thread-safe
    (check-then-set on _instances) and model loads must be serialized. The
    translator is resolved here too — cheap (just its httpx client) — so the
    concurrent translate path never touches registry.get and the lazily-created
    client is ready before any concurrent request uses it.
    """
    detector = registry.get("detector", plan.detector, device=state.resolve_device_for(plan.detector))
    recognizer = registry.get("recognizer", plan.recognizer, device=state.resolve_device_for(plan.recognizer))
    translator = registry.get("translator", plan.translator)
    recognized = detect_and_recognize(
        img, detector=detector, recognizer=recognizer,
        src=plan.src, opt_detect=plan.opt_detect, opt_recognize=plan.opt_recognize,
    )
    return recognized, translator


def _translate_sync(recognized, translator, plan: RunPlan):
    """Run the LLM half: batch-translate the recognized pairs -> wire result.
    Runs in the threadpool OUTSIDE the GPU lock (ollama is a separate process),
    so one image's translation overlaps the next image's detect+recognize."""
    return translate_regions(
        recognized, translator=translator, src=plan.src, dst=plan.dst, opt_translate=plan.opt_translate
    )


def _decode_image(contents: str) -> Image.Image:
    """base64 string -> RGB PIL image; BadImageError on anything malformed."""
    try:
        binary = base64.b64decode(contents)
        return Image.open(io.BytesIO(binary)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise BadImageError(str(exc))


async def _run_deduped(id_, compute):
    """Run the async ``compute`` once per in-flight ``id_``: concurrent requests
    for the same key await the first computation instead of repeating the model
    work. The result (or the failure) is shared with every waiter."""
    existing = state.inflight.get(id_)
    if existing is not None:
        return await existing

    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    state.inflight[id_] = fut
    try:
        result = await compute()
        fut.set_result(result)
        return result
    except Exception as exc:  # noqa: BLE001 - hand the same failure to any waiters, then re-raise
        if not fut.done():
            fut.set_exception(exc)
        raise
    finally:
        state.inflight.pop(id_, None)


async def run_page(plan: RunPlan, md5: str, contents: str) -> list:
    """Compute (or join an in-flight computation of) the page result: decode ->
    detect+recognize under the GPU lock -> translate off it -> cache. Raises
    BadImageError on a bad image; other failures propagate to the caller."""
    async def compute():
        # Timed per stage so the log shows where the time goes. lockwait = time
        # spent waiting for the GPU lock (surfaces contention when images overlap).
        logger.info("run md5=%s engines=%s %s->%s", md5[:8], plan.engines, plan.src, plan.dst)
        t0 = time.perf_counter()
        img = _decode_image(contents)
        t_dec = time.perf_counter()
        # GPU/model half under the lock (single device); translate half outside it
        # (ollama is a separate process), bounded so concurrent images don't
        # overrun the backend's parallel slots.
        async with state.gpu_lock:
            t_lock = time.perf_counter()
            recognized, translator = await run_in_threadpool(_read_sync, img, plan)
        t_det = time.perf_counter()
        async with state.translate_sem:
            result = await run_in_threadpool(_translate_sync, recognized, translator, plan)
        t_tsl = time.perf_counter()
        cache.put_run(md5, *plan.cache_key, result)
        logger.info(
            "md5=%s ok regions=%d decode=%.0f lockwait=%.0f detect+recognize=%.0f translate=%.0f total=%.0fms",
            md5[:8], len(recognized),
            (t_dec - t0) * 1000, (t_lock - t_dec) * 1000, (t_det - t_lock) * 1000,
            (t_tsl - t_det) * 1000, (t_tsl - t0) * 1000,
        )
        return result

    return await _run_deduped((md5, *plan.cache_key), compute)
