"""Selection endpoints (POST): choose the per-role engines and languages, and set
the per-engine compute device, recognize-pool size, and inference-gate size.

Selection is validated (name must exist / language must be known) but engines
are NOT eagerly loaded — that happens lazily on first run_pipeline.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from scanlation_sdk.context import LANGUAGES
from ..recognize_pool import recognize_pool
from ..registry import ROLE_NAMES, registry
from ..schemas import (
    SetEngineDeviceRequest,
    SetGpuConcurrencyRequest,
    SetLanguagesRequest,
    SetEnginesRequest,
    SetRecognizeConcurrencyRequest,
)
from ..state import state

router = APIRouter()

# GPU ("cuda") also covers ROCm — a ROCm torch build exposes HIP under the
# torch.cuda namespace, so ".to('cuda')" works there too.
# Format-only: accept cpu / cuda / cuda:<n>. Which indices actually exist is the
# UI's job (it only offers enumerated GPUs) and pick_device's (range-check + fall
# back at load), so this route stays torch-free and testable without a GPU.
_DEVICE_RE = re.compile(r"^(cpu|cuda(:\d+)?)$")


@router.post("/set_engines/")
def set_engines(req: SetEnginesRequest) -> dict:
    for role in ROLE_NAMES:
        name = getattr(req, role)
        if name and not registry.has(role, name):
            raise HTTPException(status_code=400, detail=f"unknown {role}: {name}")
    state.set_engines(req.detector, req.recognizer, req.translator)
    return {}


@router.post("/set_languages/")
def set_languages(req: SetLanguagesRequest) -> dict:
    for code in (req.lang_src, req.lang_dst):
        if code not in LANGUAGES:
            raise HTTPException(status_code=400, detail=f"unknown language: {code}")
    state.set_languages(req.lang_src, req.lang_dst)
    return {}


def _apply_engine_device(engine: str, dev: str | None) -> None:
    """Persist the device + drop the engine's cached instance so its next request
    reloads on it. Holds ``detect_lock`` so an in-flight DETECT — which runs OFF the
    gpu_gate now, so the writer around this doesn't exclude it — can't have the
    detector torn out mid-forward. Runs in a threadpool (detect can take ~270ms, so
    acquiring the lock must not block the event loop)."""
    with state.detect_lock:
        state.set_engine_device(engine, dev)
        for role in ROLE_NAMES:
            if registry.has(role, engine):
                registry.unload_one(role, engine)
        recognize_pool.invalidate(engine)  # rebuild on the new device (no-op if not the pooled engine)


@router.post("/set_engine_device/")
async def set_engine_device(req: SetEngineDeviceRequest) -> dict:
    """Per-engine compute-device override (empty device removes it -> the engine's
    DEFAULT_DEVICE). On a real change, drop that engine's cached instance (under the
    gate writer for recognize + detect_lock for the off-gate detector) so its next
    request reloads on the resolved device."""
    dev = req.device.strip().lower()
    if dev and not _DEVICE_RE.match(dev):
        raise HTTPException(status_code=400, detail="device must be cpu, cuda, or cuda:<n>")
    if not any(registry.has(role, req.engine) for role in ROLE_NAMES):
        raise HTTPException(status_code=400, detail=f"unknown engine: {req.engine}")
    if (dev or None) != state.resolve_device_for(req.engine):
        async with state.gpu_gate.writer():  # exclusive vs in-flight recognize while we swap
            await run_in_threadpool(_apply_engine_device, req.engine, dev or None)
    return {"status": "success", "device": state.resolve_device_for(req.engine) or ""}


@router.post("/set_recognize_concurrency/")
async def set_recognize_concurrency(req: SetRecognizeConcurrencyRequest) -> dict:
    """Per-engine recognize worker-pool size (null resets to the global default).
    On a real change, invalidate the pool under the GPU lock so the next run rebuilds
    at the new size and no run is torn down mid-flight."""
    if not any(registry.has(role, req.engine) for role in ROLE_NAMES):
        raise HTTPException(status_code=400, detail=f"unknown engine: {req.engine}")
    new = None if req.concurrency is None else max(1, int(req.concurrency))
    if new != state.selection.recognize_concurrency.get(req.engine):
        async with state.gpu_gate.writer():  # exclusive vs all in-flight inference while we tear the pool down
            state.set_recognize_concurrency(req.engine, new)
            recognize_pool.invalidate(req.engine)
    return {"status": "success", "concurrency": state.resolve_recognize_concurrency(req.engine)}


@router.post("/set_gpu_concurrency/")
def set_gpu_concurrency(req: SetGpuConcurrencyRequest) -> dict:
    """Per-recognizer gate size — max images running the detect+recognize half at once
    (null resets to the global default). On a real change to the ACTIVE recognizer,
    rebuild the gate: in-flight inference finishes on the old gate, new requests use
    the new size. This resizes only the gate (no pool/model teardown), so it's the
    same runtime swap as translate_sem — no writer/drain needed."""
    if not any(registry.has(role, req.engine) for role in ROLE_NAMES):
        raise HTTPException(status_code=400, detail=f"unknown engine: {req.engine}")
    new = None if req.concurrency is None else max(1, int(req.concurrency))
    if new != state.selection.gpu_concurrency.get(req.engine):
        state.set_gpu_concurrency(req.engine, new)
        if req.engine == state.selection.recognizer:
            state.rebuild_gpu_gate()
    return {"status": "success", "concurrency": state.resolve_gpu_concurrency(req.engine)}
