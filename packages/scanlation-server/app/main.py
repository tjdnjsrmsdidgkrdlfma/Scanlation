"""FastAPI app factory.

CORS is permissive: the browser content script issues cross-origin requests to
this server from arbitrary manga sites, so every origin must be allowed (the old
Django stack left this commented-out, which is why it needed extension privileges
to work). No CSRF (FastAPI has none; all POSTs are open).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings
from .routes import admin, handshake, plugins, run, settings_routes

# Admin page assets ship with the code (not the data volume), so resolve them
# relative to this package, never to SCANLATION_BASE_DIR.
WEB_DIR = Path(__file__).resolve().parent / "web"


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="scanlation-server", version="0.1.0", lifespan=lifespan)
    # Order matters: add auth first, CORS last, so CORS ends up OUTERMOST and its
    # headers are attached even to the 401s (the browser/extension can read them).
    app.add_middleware(BaseHTTPMiddleware, dispatch=_require_token)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # cannot combine credentials with "*"
        allow_methods=["*"],
        allow_headers=["*"],
    )
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
