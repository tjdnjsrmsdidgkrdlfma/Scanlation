"""Plugin listing/management.

/get_plugin_data/ lists every discovered engine (deduped by name). v1 ships all
engines built-in, so /manage_plugins/ is a success no-op (runtime pip install of
plugins is out of v1 scope).
"""
from __future__ import annotations

from fastapi import APIRouter

from .. import __version__
from ..registry import registry
from ..schemas import ManagePluginsRequest

router = APIRouter()


@router.get("/get_plugin_data/")
def get_plugin_data() -> dict:
    resp: dict = {}
    for role, mapping in registry.all_classes().items():
        for name, cls in mapping.items():
            if name in resp:  # dedupe shared names (e.g. 'dummy' in every role)
                resp[name].setdefault("roles", []).append(role)
                continue
            resp[name] = {
                "homepage": getattr(cls, "homepage", None),
                "warning": getattr(cls, "warning", None),
                "description": getattr(cls, "description", ""),
                "version": __version__,
                "installed": True,
                "roles": [role],
            }
    return resp


@router.post("/manage_plugins/")
def manage_plugins(req: ManagePluginsRequest) -> dict:
    # v1 no-op stub: all engines are built-in.
    return {"status": "success"}
