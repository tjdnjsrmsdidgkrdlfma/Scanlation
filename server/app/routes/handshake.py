"""GET / — handshake. Lightweight: lists languages/engines, loads no models.

Keys are byte-for-byte what the ocr_extension client reads (verified against
ocr_translate views.handshake).
"""
from __future__ import annotations

from fastapi import APIRouter

from .. import __version_array__
from ..config import LANGUAGES
from ..registry import registry
from ..state import state

router = APIRouter()


def handshake_payload() -> dict:
    sel = state.selection
    iso = list(LANGUAGES.keys())
    return {
        "version": __version_array__,
        "Languages": iso,
        "Languages_src": iso,
        "Languages_dst": iso,
        "Languages_hr": [LANGUAGES[k] for k in iso],
        "BOXModels": registry.names("detector"),
        "OCRModels": registry.names("recognizer"),
        "TSLModels": registry.names("translator"),
        "box_selected": sel.detector,
        "ocr_selected": sel.recognizer,
        "tsl_selected": sel.translator,
        "lang_src": sel.lang_src,
        "lang_dst": sel.lang_dst,
    }


@router.get("/")
def handshake() -> dict:
    return handshake_payload()
