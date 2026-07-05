"""Selection endpoints: /set_engines/, /set_languages/, /set_engine_device/.

Selection is validated (name must exist / language must be known) but engines
are NOT eagerly loaded — that happens lazily on first run_pipeline.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from scanlation_sdk.context import LANGUAGES
from ..registry import ROLE_NAMES, registry
from ..schemas import SetEngineDeviceRequest, SetLanguagesRequest, SetEnginesRequest
from ..state import state

router = APIRouter()

# GPU ("cuda") also covers ROCm — a ROCm torch build exposes HIP under the
# torch.cuda namespace, so ".to('cuda')" works there too. dml is gone with ctd.
DEVICES = ("cpu", "cuda")


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


@router.post("/set_engine_device/")
async def set_engine_device(req: SetEngineDeviceRequest) -> dict:
    """Per-engine compute-device override (empty device removes it -> the engine's
    DEFAULT_DEVICE). On a real change, drop that engine's cached instance under the
    GPU lock so its next request reloads on the resolved device."""
    dev = req.device.strip().lower()
    if dev and dev not in DEVICES:
        raise HTTPException(status_code=400, detail=f"device must be one of {DEVICES}")
    if not any(registry.has(role, req.engine) for role in ROLE_NAMES):
        raise HTTPException(status_code=400, detail=f"unknown engine: {req.engine}")
    if (dev or None) != state.resolve_device_for(req.engine):
        async with state.gpu_lock:      # no inference mid-flight while we swap
            state.set_engine_device(req.engine, dev or None)
            for role in ROLE_NAMES:
                if registry.has(role, req.engine):
                    registry.unload_one(role, req.engine)
    return {"status": "success", "device": state.resolve_device_for(req.engine) or ""}
