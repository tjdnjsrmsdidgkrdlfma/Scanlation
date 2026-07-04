"""Plugin management.

``/install_plugins/`` installs a plugin: pip-install its package into the plugins
volume if missing (so its heavy backend deps land only now), then download its
weights. ``/install_plugin_stream/`` does the same for one plugin but streams the
pip log and weights-download progress live (NDJSON) so /admin can show it.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..plugins_install import install_plugin, install_plugin_events
from ..schemas import InstallPluginStreamRequest, InstallPluginsRequest

router = APIRouter()


@router.post("/install_plugins/")
def install_plugins(req: InstallPluginsRequest) -> dict:
    """Install plugins — the explicit one-click action backing the admin plugin
    tab. ``{plugins: {name: true}}`` pip-installs the package (if not already) and
    downloads its weights; ``false`` is a no-op (uninstall not in scope)."""
    try:
        for name, want in req.plugins.items():
            if not want:
                continue
            install_plugin(name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)[:200])
    return {"status": "success"}


@router.post("/install_plugin_stream/")
def install_plugin_stream(req: InstallPluginStreamRequest) -> StreamingResponse:
    """One-plugin install with live progress. Streams newline-delimited JSON events
    (phase / log / done / error) as the package installs and the weights download,
    so /admin can render a live log. Failures arrive as an ``error`` event inside
    the stream (the response itself is 200 once streaming starts)."""
    def gen():
        try:
            for ev in install_plugin_events(req.name):
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as exc:  # noqa: BLE001 - last-resort guard; worker already traps most
            yield json.dumps({"event": "error", "message": str(exc)[:200]}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
