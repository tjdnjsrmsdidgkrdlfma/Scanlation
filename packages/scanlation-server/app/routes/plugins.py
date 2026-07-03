"""Plugin management.

``/manage_plugins/`` installs an engine: pip-install its package into the plugins
volume if missing (so its heavy backend deps land only now), then download its
weights.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..plugins_install import install_engine
from ..schemas import ManagePluginsRequest

router = APIRouter()


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
