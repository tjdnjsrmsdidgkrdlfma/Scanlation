"""Selection endpoints: /set_models/, /set_lang/.

Selection is validated (name must exist / language must be known) but engines
are NOT eagerly loaded — that happens lazily on first run_ocrtsl.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from scanlation_sdk.context import LANGUAGES
from ..registry import ROLE_NAMES, registry
from ..schemas import SetLangRequest, SetModelsRequest
from ..state import state

router = APIRouter()


@router.post("/set_models/")
def set_models(req: SetModelsRequest) -> dict:
    for role in ROLE_NAMES:
        name = getattr(req, role)
        if name and not registry.has(role, name):
            raise HTTPException(status_code=400, detail=f"unknown {role}: {name}")
    state.set_models(req.detector, req.recognizer, req.translator)
    return {}


@router.post("/set_lang/")
def set_lang(req: SetLangRequest) -> dict:
    for code in (req.lang_src, req.lang_dst):
        if code not in LANGUAGES:
            raise HTTPException(status_code=400, detail=f"unknown language: {code}")
    state.set_langs(req.lang_src, req.lang_dst)
    return {}
