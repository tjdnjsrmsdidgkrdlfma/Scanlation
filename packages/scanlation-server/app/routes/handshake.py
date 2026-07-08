"""GET / — handshake. Lightweight: lists languages/engines, loads no models.

The role names and result-item keys it speaks are the wire vocabulary defined in
``app.schemas``.
"""
from __future__ import annotations

from fastapi import APIRouter

from .. import __version_array__
from scanlation_sdk.context import LANGUAGES
from ..engine_meta import engine_label
from ..registry import registry
from ..state import state

router = APIRouter()


def _hr(role: str, names: list[str]) -> list[str]:
    """Human-readable engine labels (display_name, fallback to the id) parallel to
    the id list — the popup/extension shows these instead of raw ids."""
    return [engine_label(registry.get_class(role, n), n) for n in names]


def handshake_payload() -> dict:
    sel = state.selection
    iso = list(LANGUAGES.keys())
    dets = registry.names("detector")
    recs = registry.names("recognizer")
    translators = registry.names("translator")
    return {
        "version": __version_array__,
        "Languages": iso,
        "Languages_src": iso,
        "Languages_dst": iso,
        "Languages_hr": [LANGUAGES[k] for k in iso],
        "detectors": dets,
        "recognizers": recs,
        "translators": translators,
        "detectors_hr": _hr("detector", dets),
        "recognizers_hr": _hr("recognizer", recs),
        "translators_hr": _hr("translator", translators),
        "detector_selected": sel.detector,
        "recognizer_selected": sel.recognizer,
        "translator_selected": sel.translator,
        "lang_src": sel.lang_src,
        "lang_dst": sel.lang_dst,
        "min_image_dim": sel.min_image_dim,   # extension image filter (shorter-side px)
    }


@router.get("/")
def handshake() -> dict:
    return handshake_payload()
