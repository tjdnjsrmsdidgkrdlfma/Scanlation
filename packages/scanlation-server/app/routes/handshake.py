"""GET / — handshake. Lightweight: lists languages/engines, loads no models.

Engine roles use the same names end-to-end (detector/recognizer/translator);
the old ocr_extension BOX/OCR/TSL wire vocabulary was dropped.
"""
from __future__ import annotations

from fastapi import APIRouter

from .. import __version_array__
from scanlation_sdk.context import LANGUAGES
from ..registry import registry
from ..state import state

router = APIRouter()


def _hr(role: str, names: list[str]) -> list[str]:
    """Human-readable engine labels (display_name, fallback to the id) parallel to
    the id list — the popup/extension shows these instead of raw ids."""
    return [getattr(registry.get_class(role, n), "display_name", None) or n for n in names]


def handshake_payload() -> dict:
    sel = state.selection
    iso = list(LANGUAGES.keys())
    dets = registry.names("detector")
    recs = registry.names("recognizer")
    tsls = registry.names("translator")
    return {
        "version": __version_array__,
        "Languages": iso,
        "Languages_src": iso,
        "Languages_dst": iso,
        "Languages_hr": [LANGUAGES[k] for k in iso],
        "detectors": dets,
        "recognizers": recs,
        "translators": tsls,
        "detectors_hr": _hr("detector", dets),
        "recognizers_hr": _hr("recognizer", recs),
        "translators_hr": _hr("translator", tsls),
        "detector_selected": sel.detector,
        "recognizer_selected": sel.recognizer,
        "translator_selected": sel.translator,
        "lang_src": sel.lang_src,
        "lang_dst": sel.lang_dst,
    }


@router.get("/")
def handshake() -> dict:
    return handshake_payload()
