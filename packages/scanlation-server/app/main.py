"""FastAPI app factory.

CORS is permissive: the browser content script issues cross-origin requests to
this server from arbitrary manga sites, so every origin must be allowed (the old
Django stack left this commented-out, which is why it needed extension privileges
to work). No CSRF (FastAPI has none; all POSTs are open).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .routes import admin, handshake, plugins, run, settings_routes

# Admin page assets ship with the code (not the data volume), so resolve them
# relative to this package, never to SCANLATION_BASE_DIR.
WEB_DIR = Path(__file__).resolve().parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="scanlation-server", version="0.1.0", lifespan=lifespan)
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
