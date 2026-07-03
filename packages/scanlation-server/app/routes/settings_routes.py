"""Selection endpoints: /set_engines/, /set_languages/, /set_device/.

Selection is validated (name must exist / language must be known) but engines
are NOT eagerly loaded — that happens lazily on first run_pipeline.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from scanlation_sdk.context import LANGUAGES
from ..registry import ROLE_NAMES, registry
from ..schemas import SetDeviceRequest, SetLanguagesRequest, SetEnginesRequest
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


@router.post("/set_device/")
async def set_device(req: SetDeviceRequest) -> dict:
    """Persist the compute device (cpu/cuda) for detector + recognizer. On a real
    change, drop cached engine instances under the GPU lock so the next request
    reloads them on the new device; unchanged = no-op (no reload)."""
    dev = req.device.strip().lower()
    if dev not in DEVICES:
        raise HTTPException(status_code=400, detail=f"device must be one of {DEVICES}")
    if dev != state.selection.device:
        async with state.gpu_lock:      # no inference mid-flight while we swap
            state.set_device(dev)       # persist + update shared context.device
            registry.unload_all()       # next get() reloads on the new device
    return {"status": "success", "device": state.selection.device}
