"""OCR+translate endpoints: /run_ocrtsl/, /run_tsl/, /get_trans/.

run_ocrtsl implements the verified lazy/work flow:
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
from ..pipeline import run_pipeline, translate_text
from ..registry import registry
from ..schemas import RunOcrTslRequest, RunTslRequest
from ..state import state

router = APIRouter()


def _resolve():
    """Current (names, langs, engines-id) from selection."""
    s = state.selection
    return (
        s.detector, s.recognizer, s.translator,
        s.lang_src, s.lang_dst,
        f"{s.detector}+{s.recognizer}+{s.translator}",
    )


def _work_sync(img, det_name, rec_name, tsl_name, src, dst, opt_box, opt_ocr, opt_tsl):
    """Resolve engines (loads weights on first use) and run the pipeline.

    Runs entirely in the threadpool so model load + inference never block the
    event loop.
    """
    detector = registry.get("detector", det_name)
    recognizer = registry.get("recognizer", rec_name)
    translator = registry.get("translator", tsl_name)
    return run_pipeline(
        img, detector=detector, recognizer=recognizer, translator=translator,
        src=src, dst=dst, opt_box=opt_box, opt_ocr=opt_ocr, opt_tsl=opt_tsl,
    )


@router.post("/run_ocrtsl/")
async def run_ocrtsl(req: RunOcrTslRequest) -> dict:
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

    # Dedupe concurrent identical requests onto one computation.
    id_ = (req.md5, src, dst, engines, oh)
    existing = state.inflight.get(id_)
    if existing is not None:
        return {"result": await existing}

    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    state.inflight[id_] = fut
    try:
        try:
            binary = base64.b64decode(req.contents)
            img = Image.open(io.BytesIO(binary)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"bad image: {exc}")

        async with state.gpu_lock:
            result = await run_in_threadpool(
                _work_sync, img, det, rec, tsl, src, dst, opt_box, opt_ocr, opt_tsl
            )
        cache.put_run(req.md5, src, dst, engines, oh, result)
        fut.set_result(result)
        return {"result": result}
    except HTTPException as exc:
        if not fut.done():
            fut.set_exception(exc)
        raise
    except Exception as exc:  # noqa: BLE001
        if not fut.done():
            fut.set_exception(exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        state.inflight.pop(id_, None)


@router.post("/run_tsl/")
async def run_tsl(req: RunTslRequest) -> dict:
    s = state.selection
    translator = registry.get("translator", s.translator)
    opt_tsl = state.translator_options(s.translator, None)
    text = await run_in_threadpool(
        translate_text, req.text, s.lang_src, s.lang_dst, translator, opt_tsl
    )
    return {"text": text}


@router.get("/get_trans/")
async def get_trans(text: str, lang_src: str | None = None, lang_dst: str | None = None) -> dict:
    s = state.selection
    src = lang_src or s.lang_src
    dst = lang_dst or s.lang_dst
    return {"translations": cache.get_translations(text, src, dst)}
