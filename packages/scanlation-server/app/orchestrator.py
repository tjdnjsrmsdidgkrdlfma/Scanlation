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
from .pipeline import detect_and_recognize, recognized_to_result, translate_regions
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
    engines (loads weights on first use); returns (recognized pairs, translator,
    sub-timing) where sub-timing is {detect_ms, recognize_ms} from the two halves.

    Runs in the threadpool UNDER the GPU lock: registry.get is not thread-safe
    (check-then-set on _instances) and model loads must be serialized. The
    translator is resolved here too — cheap (just its httpx client) — so the
    concurrent translate path never touches registry.get and the lazily-created
    client is ready before any concurrent request uses it.
    """
    detector = registry.get("detector", plan.detector)
    recognizer = registry.get("recognizer", plan.recognizer)
    translator = registry.get("translator", plan.translator)
    recognized, sub = detect_and_recognize(
        img, detector=detector, recognizer=recognizer,
        src=plan.src, opt_detect=plan.opt_detect, opt_recognize=plan.opt_recognize,
    )
    return recognized, translator, sub


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
            # Mark it read. We re-raise below, and any waiter awaiting `fut` still gets
            # the exception -- but with no waiter nobody would ever read it, and asyncio
            # dumps "Future exception was never retrieved" when it finalizes the future.
            fut.exception()
        raise
    finally:
        state.inflight.pop(id_, None)


async def run_page(plan: RunPlan, md5: str, contents: str, *, skip_translate: bool = False) -> tuple[list, dict]:
    """Compute (or join an in-flight computation of) the page result: decode ->
    detect+recognize under the GPU lock -> translate off it -> cache. Returns
    ``(result, timing)`` where timing is the per-stage ms breakdown (also logged).
    Raises BadImageError on a bad image; other failures propagate to the caller.

    ``skip_translate`` is a recognize-only mode for benchmarks: it emits the source
    with an empty destination, skips the LLM half (semwait/translate stay 0), and does
    NOT cache — a translation-less result must never shadow a real one."""
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
            recognized, translator, sub = await run_in_threadpool(_read_sync, img, plan)
        t_det = time.perf_counter()
        if skip_translate:
            # recognize-only: source with no translation; leave the translate spans at 0
            # and don't cache (a translation-less result must not shadow a real one).
            result = recognized_to_result(recognized)
            t_sem = t_tsl = t_det
        else:
            async with state.translate_sem:
                t_sem = time.perf_counter()  # sem acquired: split our queue-wait from the actual call
                result = await run_in_threadpool(_translate_sync, recognized, translator, plan)
            t_tsl = time.perf_counter()
            cache.put_run(md5, *plan.cache_key, result)
        # semwait = time queued on translate_sem (our backpressure); translate = the
        # actual backend call (ollama generate + any ollama-side queue). Splitting them
        # tells whether OUR limit or the BACKEND is the bottleneck when translates pile up.
        logger.info(
            "md5=%s ok regions=%d decode=%.0f lockwait=%.0f detect+recognize=%.0f "
            "semwait=%.0f translate=%.0f total=%.0fms",
            md5[:8], len(recognized),
            (t_dec - t0) * 1000, (t_lock - t_dec) * 1000, (t_det - t_lock) * 1000,
            (t_sem - t_det) * 1000, (t_tsl - t_sem) * 1000, (t_tsl - t0) * 1000,
        )
        # Same spans as the log line, returned so the /run_pipeline/ response can carry
        # them (headless tools/reports read this; the extension ignores the extra key).
        timing = {
            "decode_ms": round((t_dec - t0) * 1000, 1),
            "lockwait_ms": round((t_lock - t_dec) * 1000, 1),
            # detect_recognize_ms is the whole GPU half (t_det - t_lock): it also covers
            # engine resolve/first-load + threadpool handoff, so it's >= detect + recognize.
            "detect_recognize_ms": round((t_det - t_lock) * 1000, 1),
            "detect_ms": sub["detect_ms"],
            "recognize_ms": sub["recognize_ms"],
            "semwait_ms": round((t_sem - t_det) * 1000, 1),
            "translate_ms": round((t_tsl - t_sem) * 1000, 1),
            "total_ms": round((t_tsl - t0) * 1000, 1),
            "regions": len(recognized),
        }
        return result, timing

    # skip_translate in the dedup key so a recognize-only run never joins (or is joined
    # by) a full run for the same image — they compute different results.
    return await _run_deduped((md5, *plan.cache_key, skip_translate), compute)
