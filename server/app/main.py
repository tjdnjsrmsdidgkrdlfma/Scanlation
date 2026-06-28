"""FastAPI app factory.

CORS is permissive: the browser content script issues cross-origin requests to
this server from arbitrary manga sites, so every origin must be allowed (the old
Django stack left this commented-out, which is why it needed extension privileges
to work). No CSRF (FastAPI has none; all POSTs are open).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import handshake, manual, plugins, run, settings_routes


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
    app.include_router(manual.router)
    app.include_router(settings_routes.router)
    app.include_router(plugins.router)
    return app


app = create_app()
