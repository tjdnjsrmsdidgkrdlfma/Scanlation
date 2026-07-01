"""Plugin listing/management.

/get_plugin_data/ lists every discovered engine (deduped by name). v1 ships all
engines built-in, so /manage_plugins/ is a success no-op (runtime pip install of
plugins is out of v1 scope).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

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
                resp[name]["roles"].append(role)
                continue
            try:  # cheap __init__ only (no load); is_installed checks the filesystem/cache
                installed = cls().is_installed()
            except Exception:  # noqa: BLE001
                installed = False
            resp[name] = {
                "homepage": getattr(cls, "homepage", None),
                "warning": getattr(cls, "warning", None),
                "description": getattr(cls, "description", ""),
                "version": __version__,
                "installed": installed,
                "roles": [role],
            }
    return resp


@router.post("/manage_plugins/")
def manage_plugins(req: ManagePluginsRequest) -> dict:
    """Install an engine's resources — the explicit one-click action backing the
    popup's plugin tab. ``{plugins: {name: true}}`` installs (downloads weights);
    ``false`` is a no-op (uninstall not in v1)."""
    try:
        for name, want in req.plugins.items():
            if not want:
                continue
            targets = [
                cls
                for mapping in registry.all_classes().values()
                for n, cls in mapping.items()
                if n == name
            ]
            if not targets:
                raise ValueError(f"unknown plugin: {name}")
            for cls in targets:
                cls().install()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)[:200])
    return {"status": "success"}
