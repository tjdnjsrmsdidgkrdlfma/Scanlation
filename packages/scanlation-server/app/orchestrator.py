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
from .pipeline import detect_regions, recognize_regions, recognized_to_result, translate_regions
from .recognize_pool import recognize_pool
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


def _detect_sync(img, plan: RunPlan):
    """DETECT stage (CPU) — run OFF the recognize gate. Resolves the detector +
    translator (both off-GPU: the translator is an ollama HTTP client and must not
    hold a recognize permit while it instantiates). ``detect_lock`` spans BOTH the
    detector resolve and the forward, so an /admin device change or idle-unload — which
    can no longer be excluded by the gate WRITER now that detect is off the gate —
    can't drop the detector mid-forward (see set_engine_device / idle_unload). Returns
    (regions, translator, detect_ms). registry.get is self-thread-safe."""
    with state.detect_lock:
        detector = registry.get("detector", plan.detector)
        regions, detect_ms = detect_regions(
            img, detector=detector, src=plan.src,
            opt_detect=plan.opt_detect, detect_lock=None,  # already held above
        )
    translator = registry.get("translator", plan.translator)
    return regions, translator, detect_ms


def _recognize_sync(img, regions, plan: RunPlan):
    """RECOGNIZE stage (GPU) — run UNDER the gate reader (up to K images' crops share
    the pool). Takes the worker pool when this recognizer's concurrency is >1 (a page's
    crops fan out across processes; the recognizer is NOT registry-loaded here so its
    VRAM lives only in the workers, not also in this process); otherwise the
    registry-loaded engine runs the in-process per-crop loop (the default). A pool that
    stays broken after its own rebuild+retry propagates — the request fails rather than
    doubling the VRAM by loading the model here. Returns (recognized pairs, rec-timing)
    where rec-timing is {recognize_ms, raw_regions, region_details}."""
    workers = state.resolve_recognize_concurrency(plan.recognizer)
    if workers > 1:
        recognize_pool.ensure(plan.recognizer, state.resolve_device_for(plan.recognizer), workers)
        return recognize_regions(img, regions, recognizer=None, opt_recognize=plan.opt_recognize,
                                 pool=recognize_pool, rec_name=plan.recognizer)
    recognizer = registry.get("recognizer", plan.recognizer)
    return recognize_regions(img, regions, recognizer=recognizer, opt_recognize=plan.opt_recognize)


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


def _build_timing(*, t0, t_dec, t_det, t_lock, t_rec, t_sem, t_tsl, detect_ms, recognize_ms, regions) -> dict:
    """The per-stage ms breakdown from the pipeline's perf_counter marks. detect_ms/
    recognize_ms are the pure forwards from the two sync helpers; lockwait = wait for a
    recognize gate permit (detect is off the gate); semwait = queued on translate_sem;
    translate = the actual ollama call. These spans don't sum exactly to total (residual
    = engine resolve/first-load + threadpool handoff), which is fine for a breakdown."""
    return {
        "decode_ms": round((t_dec - t0) * 1000, 1),
        "detect_ms": detect_ms,
        "lockwait_ms": round((t_lock - t_det) * 1000, 1),
        "recognize_ms": recognize_ms,
        "semwait_ms": round((t_sem - t_rec) * 1000, 1),
        "translate_ms": round((t_tsl - t_sem) * 1000, 1),
        "total_ms": round((t_tsl - t0) * 1000, 1),
        "regions": regions,
    }


def _build_region_rows(rec_sub: dict, result: list) -> list:
    """Per-crop stats rows: each region_detail plus a dest_len paired to result's
    translations in non-empty reading order (empty-recognition crops get 0)."""
    dests = iter(len(it["destination"]) for it in result)
    return [{**d, "dest_len": (next(dests) if d["source_len"] else 0)}
            for d in rec_sub.get("region_details", [])]


async def run_page(plan: RunPlan, md5: str, contents: str, *, skip_translate: bool = False) -> tuple[list, dict]:
    """Compute (or join an in-flight computation of) the page result: decode ->
    detect (off the gate, CPU) -> recognize (under the gate) -> translate (off it) ->
    cache. Returns ``(result, timing)`` — timing is the per-stage ms breakdown (also logged).
    Raises BadImageError on a bad image; other failures propagate to the caller.

    ``skip_translate`` is a recognize-only mode for benchmarks: it emits the source
    with an empty destination, skips the LLM half (semwait/translate stay 0), and does
    NOT cache — a translation-less result must never shadow a real one."""
    async def compute():
        # Timed per stage so the log shows where the time goes. lockwait = time spent
        # waiting for a gate reader permit (surfaces contention when images overlap).
        logger.info("run md5=%s engines=%s %s->%s", md5[:8], plan.engines, plan.src, plan.dst)
        t0 = time.perf_counter()
        img = _decode_image(contents)
        t_dec = time.perf_counter()
        # DETECT off the gate (CPU; serialized by detect_lock inside _detect_sync).
        # Detect no longer holds a recognize permit, so the K gate slots feed recognize
        # only — already-detected images' crops are what's resident in the pool.
        regions, translator, detect_ms = await run_in_threadpool(_detect_sync, img, plan)
        t_det = time.perf_counter()
        # RECOGNIZE under the gate (reader; up to K images' crops share the pool =
        # cross-image overlap). Translate runs off the gate (ollama is a separate
        # process), bounded so concurrent images don't overrun its parallel slots.
        async with state.gpu_gate.reader():
            t_lock = time.perf_counter()
            recognized, rec_sub = await run_in_threadpool(_recognize_sync, img, regions, plan)
        t_rec = time.perf_counter()
        if skip_translate:
            # recognize-only: source with no translation; leave the translate spans at 0
            # and don't cache (a translation-less result must not shadow a real one).
            result = recognized_to_result(recognized)
            t_sem = t_tsl = t_rec
        else:
            async with state.translate_sem:
                t_sem = time.perf_counter()  # sem acquired: split our queue-wait from the actual call
                result = await run_in_threadpool(_translate_sync, recognized, translator, plan)
            t_tsl = time.perf_counter()
            cache.put_run(md5, *plan.cache_key, result)
        # Per-stage spans, returned so the /run_pipeline/ response can carry them
        # (headless tools/reports read this; the extension ignores the extra key).
        timing = _build_timing(
            t0=t0, t_dec=t_dec, t_det=t_det, t_lock=t_lock, t_rec=t_rec, t_sem=t_sem,
            t_tsl=t_tsl, detect_ms=detect_ms, recognize_ms=rec_sub["recognize_ms"],
            regions=len(recognized),
        )
        logger.info(
            "md5=%s ok regions=%d decode=%.0f detect=%.0f lockwait=%.0f recognize=%.0f "
            "semwait=%.0f translate=%.0f total=%.0fms",
            md5[:8], len(recognized),
            timing["decode_ms"], timing["detect_ms"], timing["lockwait_ms"], timing["recognize_ms"],
            timing["semwait_ms"], timing["translate_ms"], timing["total_ms"],
        )
        # Persist raw per-page + per-crop stats. region_details/raw_regions ride `rec_sub`
        # (the recognize timing); skip_translate is marked and default-filtered out of
        # the summary.
        region_rows = _build_region_rows(rec_sub, result)
        cache.record_stats(
            page={"engines": plan.engines, "src": plan.src, "dst": plan.dst, "md5": md5,
                  "regions": timing["regions"], "raw_regions": rec_sub.get("raw_regions"),
                  **{k: timing[k] for k in ("decode_ms", "lockwait_ms", "detect_ms",
                     "recognize_ms", "semwait_ms", "translate_ms", "total_ms")}},
            regions=region_rows, skip_translate=skip_translate,
        )
        return result, timing

    # skip_translate in the dedup key so a recognize-only run never joins (or is joined
    # by) a full run for the same image — they compute different results.
    return await _run_deduped((md5, *plan.cache_key, skip_translate), compute)
