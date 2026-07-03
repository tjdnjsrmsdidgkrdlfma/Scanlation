"""OCR+translate endpoint: /run_pipeline/.

run_pipeline implements the verified lazy/work flow:
  * lazy  : client POSTs {md5, options}   -> cache hit returns result, miss = 404
  * work  : client POSTs {md5, contents}  -> md5(base64) verified, pipeline runs
md5 is computed over the base64 *string* (not raw bytes) — mismatch => 400.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io

from fastapi import APIRouter, HTTPException
from PIL import Image
from starlette.concurrency import run_in_threadpool

from ..cache import cache, opt_hash
from ..pipeline import detect_and_recognize, translate_regions
from ..registry import registry
from ..schemas import RunPipelineRequest
from ..state import state

router = APIRouter()


def _require(role: str, name: str) -> None:
    """400 if a role has no engine installed/selected (the core ships none)."""
    if not name or not registry.has(role, name):
        raise HTTPException(
            status_code=400,
            detail=f"no {role} engine installed — install and select one in /admin",
        )


def _resolve():
    """Current (names, langs, engines-id) from selection."""
    sel = state.selection
    return (
        sel.detector, sel.recognizer, sel.translator,
        sel.lang_src, sel.lang_dst,
        f"{sel.detector}+{sel.recognizer}+{sel.translator}",
    )


def _detect_sync(img, det_name, rec_name, tsl_name, src, opt_box, opt_ocr):
    """Resolve all three engines (loads weights on first use) and run the GPU/model
    half: detect + recognize. Returns (recognized pairs, translator instance).

    Runs in the threadpool UNDER the GPU lock: registry.get is not thread-safe
    (check-then-set on _instances) and model loads must be serialized. The
    translator is resolved here too — cheap (just its httpx client) — so the
    concurrent translate path never touches registry.get and the lazily-created
    client is ready before any concurrent request uses it.
    """
    detector = registry.get("detector", det_name)
    recognizer = registry.get("recognizer", rec_name)
    translator = registry.get("translator", tsl_name)
    recognized = detect_and_recognize(
        img, detector=detector, recognizer=recognizer, src=src, opt_box=opt_box, opt_ocr=opt_ocr
    )
    return recognized, translator


def _translate_sync(recognized, translator, src, dst, opt_tsl):
    """Run the LLM half: batch-translate the recognized pairs -> wire result.
    Runs in the threadpool OUTSIDE the GPU lock (ollama is a separate process),
    so one image's translation overlaps the next image's detect+recognize."""
    return translate_regions(recognized, translator=translator, src=src, dst=dst, opt_tsl=opt_tsl)


def _decode_image(contents: str) -> Image.Image:
    """base64 string -> RGB PIL image; 400 on anything malformed."""
    try:
        binary = base64.b64decode(contents)
        return Image.open(io.BytesIO(binary)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad image: {exc}")


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


@router.post("/run_pipeline/")
async def run_pipeline(req: RunPipelineRequest) -> dict:
    det, rec, tsl, src, dst, engines = _resolve()
    opt_box = state.options_for(det, req.options)
    opt_ocr = state.options_for(rec, req.options)
    opt_tsl = state.translator_options(tsl, req.options)
    oh = opt_hash(opt_box, opt_ocr, opt_tsl)

    # --- lazy: md5 only, no contents ---
    if req.contents is None:
        if req.force:
            raise HTTPException(status_code=400, detail="Cannot force ocr without contents")
        cached = cache.get_run(req.md5, src, dst, engines, oh)
        if cached is None:
            raise HTTPException(status_code=404, detail="cache miss")  # non-2xx -> client sends work
        return {"result": cached}

    # --- work: verify md5 over the base64 string ---
    if req.md5 != hashlib.md5(req.contents.encode("utf-8")).hexdigest():
        raise HTTPException(status_code=400, detail="md5 mismatch")

    if not req.force:
        cached = cache.get_run(req.md5, src, dst, engines, oh)
        if cached is not None:
            return {"result": cached}

    # Need a real engine per role to run (cache miss) -> 400 if any is missing.
    _require("detector", det)
    _require("recognizer", rec)
    _require("translator", tsl)

    async def _compute():
        img = _decode_image(req.contents)
        # GPU/model half under the lock (single device); translate half outside it
        # (ollama is a separate process), bounded so concurrent images don't
        # overrun the backend's parallel slots.
        async with state.gpu_lock:
            recognized, translator = await run_in_threadpool(
                _detect_sync, img, det, rec, tsl, src, opt_box, opt_ocr
            )
        async with state.translate_sem:
            result = await run_in_threadpool(
                _translate_sync, recognized, translator, src, dst, opt_tsl
            )
        cache.put_run(req.md5, src, dst, engines, oh, result)
        return result

    try:
        result = await _run_deduped((req.md5, src, dst, engines, oh), _compute)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return {"result": result}
