"""FastAPI app factory.

CORS is permissive: the browser content script issues cross-origin requests to
this server from arbitrary manga sites, so every origin must be allowed (the old
Django stack left this commented-out, which is why it needed extension privileges
to work). No CSRF (FastAPI has none; all POSTs are open).
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings
from .logconfig import configure_logging
from .routes import admin, handshake, plugins, run, settings_routes

# Admin page assets ship with the code (not the data volume), so resolve them
# relative to this package, never to SCANLATION_BASE_DIR.
WEB_DIR = Path(__file__).resolve().parent / "web"

# Access log — replaces uvicorn's (silenced in logconfig) with a timestamped line
# that also carries the request duration, so it's clear where time goes.
_http = logging.getLogger("scanlation.http")


async def _require_token(request: Request, call_next):
    """Gate every API request behind SCANLATION_AUTH_TOKEN (sent as X-Auth-Token).

    Read live so tests can toggle it. No token configured -> open (local/dev).
    Exemptions: OPTIONS (CORS preflight carries no header) and the /admin static
    shell (must load so a token can be entered; its API calls are still gated).
    """
    token = settings.auth_token
    if token and request.method != "OPTIONS" and not request.url.path.startswith("/admin"):
        if request.headers.get("X-Auth-Token") != token:
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


async def _log_requests(request: Request, call_next):
    """Access log: one timestamped ``METHOD PATH -> STATUS Nms`` line per request.
    Skips OPTIONS (CORS preflight) noise. Added outermost so the duration covers
    the full stack and the final status (incl. 401/500) is what's logged."""
    if request.method == "OPTIONS":
        return await call_next(request)
    t0 = time.perf_counter()
    resp = await call_next(request)
    _http.info(
        "%s %s -> %d %.0fms",
        request.method, request.url.path, resp.status_code, (time.perf_counter() - t0) * 1000,
    )
    return resp


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    # The persisted 동작-tab verbose toggle wins over the env default after a restart.
    from .logconfig import apply_verbose
    from .state import state
    apply_verbose(state.selection.verbose_log)
    # Wire the per-engine device resolver here (the composition root) so the
    # registry stays free of a state import. Tools/tests leave it None -> engines
    # load on their DEFAULT_DEVICE.
    from .registry import registry
    registry.device_resolver = state.resolve_device_for
    settings.ensure_dirs()
    # Periodic idle-unload of local torch engines (VRAM reclaim between sessions);
    # reads state.selection.model_idle_unload_minutes each pass. Cancelled on shutdown.
    from .idle_unload import sweep_loop
    sweep_task = asyncio.create_task(sweep_loop())
    try:
        yield
    finally:
        sweep_task.cancel()
        try:
            await sweep_task
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    app = FastAPI(title="scanlation-server", version="0.1.0", lifespan=lifespan)
    # Order matters: add auth first, CORS next (so CORS wraps auth and its headers
    # attach even to 401s), timing LAST so it's OUTERMOST — its duration covers the
    # whole stack and it logs the final status.
    app.add_middleware(BaseHTTPMiddleware, dispatch=_require_token)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # cannot combine credentials with "*"
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(BaseHTTPMiddleware, dispatch=_log_requests)
    app.include_router(handshake.router)
    app.include_router(run.router)
    app.include_router(settings_routes.router)
    app.include_router(plugins.router)
    app.include_router(admin.router)
    # Admin SPA at /admin (StaticFiles html=True serves index.html on /admin/).
    if WEB_DIR.is_dir():
        app.mount("/admin", StaticFiles(directory=str(WEB_DIR), html=True), name="admin")
    return app


app = create_app()
