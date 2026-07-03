"""Turning engine *classes* into the JSON the listings/handshake/UI consume —
without instantiating them for a real ``load()``.

The handshake and admin routes all need to read an ``EngineBase``
subclass's metadata (display name, option schema) and its cheap install status.
This is the one place that knows how, so a change to the wire shape lands once.
"""
from __future__ import annotations

from .registry import registry  # noqa: F401 (kept for callers importing from here)


def serialize_schema(cls) -> dict:
    """``OPTION_SCHEMA`` with each ``type`` object rendered as its name
    (``int``/``float``/``bool``/``str``) so it survives JSON — the admin form and
    the extension popup branch on exactly these strings."""
    out: dict = {}
    for opt, spec in getattr(cls, "OPTION_SCHEMA", {}).items():
        spec = dict(spec)
        t = spec.get("type", str)
        spec["type"] = getattr(t, "__name__", str(t))
        out[opt] = spec
    return out


def safe_is_installed(cls) -> bool:
    """``is_installed()`` via a cheap throwaway ``__init__`` (never ``load()``);
    ``False`` if the check itself raises. Used by the admin listings."""
    try:  # cheap __init__ only (no load) — is_installed checks the filesystem/cache
        return cls().is_installed()
    except Exception:  # noqa: BLE001 - a broken engine shows as "not installed", not a 500
        return False


def engine_label(cls, name: str) -> str:
    """The human-readable engine name shown in the popup/admin (id as fallback)."""
    return getattr(cls, "display_name", None) or name


def class_meta(cls, name: str) -> dict:
    """The display metadata every engine listing reads off the class."""
    return {
        "display_name": engine_label(cls, name),
        "description": getattr(cls, "description", ""),
        "warning": getattr(cls, "warning", None),
        "homepage": getattr(cls, "homepage", None),
    }
