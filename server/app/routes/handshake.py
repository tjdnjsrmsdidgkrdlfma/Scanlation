"""GET / — handshake. Lightweight: lists languages/engines, loads no models.

Engine roles use the same names end-to-end (detector/recognizer/translator);
the old ocr_extension BOX/OCR/TSL wire vocabulary was dropped.
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
        "detectors": registry.names("detector"),
        "recognizers": registry.names("recognizer"),
        "translators": registry.names("translator"),
        "detector_selected": sel.detector,
        "recognizer_selected": sel.recognizer,
        "translator_selected": sel.translator,
        "lang_src": sel.lang_src,
        "lang_dst": sel.lang_dst,
    }


@router.get("/")
def handshake() -> dict:
    return handshake_payload()
