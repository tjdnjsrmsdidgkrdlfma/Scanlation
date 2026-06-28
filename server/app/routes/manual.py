"""POST /set_manual_translation/ — record a user override (model='manual')."""
from __future__ import annotations

from fastapi import APIRouter

from ..cache import cache
from ..schemas import SetManualRequest
from ..state import state

router = APIRouter()


@router.post("/set_manual_translation/")
def set_manual_translation(req: SetManualRequest) -> dict:
    s = state.selection
    cache.put_translation(req.text, s.lang_src, s.lang_dst, "manual", req.translation)
    return {}
