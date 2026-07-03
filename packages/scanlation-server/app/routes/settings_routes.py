"""Selection endpoints: /set_engines/, /set_languages/.

Selection is validated (name must exist / language must be known) but engines
are NOT eagerly loaded — that happens lazily on first run_pipeline.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from scanlation_sdk.context import LANGUAGES
from ..registry import ROLE_NAMES, registry
from ..schemas import SetLanguagesRequest, SetEnginesRequest
from ..state import state

router = APIRouter()


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
