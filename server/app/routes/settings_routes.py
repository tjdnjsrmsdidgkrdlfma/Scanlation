"""Selection endpoints: /set_models/, /set_lang/, /get_active_options/.

Selection is validated (name must exist / language must be known) but engines
are NOT eagerly loaded — that happens lazily on first run_ocrtsl.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..config import LANGUAGES
from ..registry import registry
from ..schemas import SetLangRequest, SetModelsRequest
from ..state import state

router = APIRouter()

# wire role name -> (registry role, schema key)
_ROLE = {
    "box_model": ("detector", "box_model_id"),
    "ocr_model": ("recognizer", "ocr_model_id"),
    "tsl_model": ("translator", "tsl_model_id"),
}


def _serialize_schema(cls) -> dict:
    out: dict = {}
    for opt, spec in getattr(cls, "OPTION_SCHEMA", {}).items():
        spec = dict(spec)
        t = spec.get("type", str)
        spec["type"] = getattr(t, "__name__", str(t))
        out[opt] = spec
    return out


@router.post("/set_models/")
def set_models(req: SetModelsRequest) -> dict:
    pairs = [
        ("detector", req.box_model_id),
        ("recognizer", req.ocr_model_id),
        ("translator", req.tsl_model_id),
    ]
    for role, name in pairs:
        if name and not registry.has(role, name):
            raise HTTPException(status_code=400, detail=f"unknown {role}: {name}")
    state.set_models(req.box_model_id, req.ocr_model_id, req.tsl_model_id)
    return {}


@router.post("/set_lang/")
def set_lang(req: SetLangRequest) -> dict:
    for code in (req.lang_src, req.lang_dst):
        if code not in LANGUAGES:
            raise HTTPException(status_code=400, detail=f"unknown language: {code}")
    state.set_langs(req.lang_src, req.lang_dst)
    return {}


@router.get("/get_active_options/")
def get_active_options() -> dict:
    sel = state.selection
    selected = {
        "box_model": ("detector", sel.detector),
        "ocr_model": ("recognizer", sel.recognizer),
        "tsl_model": ("translator", sel.translator),
    }
    res: dict = {}
    for wire_key, (role, name) in selected.items():
        res[wire_key] = _serialize_schema(registry.get_class(role, name)) if registry.has(role, name) else {}
    return {"options": res}
