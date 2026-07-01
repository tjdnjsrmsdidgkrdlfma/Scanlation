"""Plugin listing/management.

``/get_plugin_data/`` lists every engine — both the ones already installed
(discovered via entry_points) and the ones merely *installable* (their source
ships in the image but the package isn't pip-installed yet). ``/manage_plugins/``
installs an engine: pip-install its package into the plugins volume if missing
(so its heavy backend deps land only now), then download its weights.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import __version__
from ..plugins_install import catalog, install_engine
from ..registry import registry
from ..schemas import ManagePluginsRequest

router = APIRouter()


@router.get("/get_plugin_data/")
def get_plugin_data() -> dict:
    resp: dict = {}
    # installed (pip-present) engines, discovered via entry_points
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
                "display_name": getattr(cls, "display_name", name),
                "homepage": getattr(cls, "homepage", None),
                "warning": getattr(cls, "warning", None),
                "description": getattr(cls, "description", ""),
                "version": __version__,
                "installed": installed,          # weights present?
                "installed_package": True,       # pip package present?
                "roles": [role],
            }
    # installable-but-not-installed engines (source shipped, package absent)
    for name, entry in catalog().items():
        if name in resp:
            continue
        resp[name] = {
            "display_name": entry.display_name,
            "homepage": None,
            "warning": None,
            "description": entry.description,
            "version": None,
            "installed": False,
            "installed_package": False,
            "roles": list(entry.roles),
        }
    return resp


@router.post("/manage_plugins/")
def manage_plugins(req: ManagePluginsRequest) -> dict:
    """Install engines — the explicit one-click action backing the admin plugin
    tab. ``{plugins: {name: true}}`` pip-installs the package (if not already) and
    downloads its weights; ``false`` is a no-op (uninstall not in scope)."""
    try:
        for name, want in req.plugins.items():
            if not want:
                continue
            install_engine(name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)[:200])
    return {"status": "success"}
