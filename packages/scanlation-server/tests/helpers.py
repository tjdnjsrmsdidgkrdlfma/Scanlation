"""Shared test helpers — no pytest, matching the hand-rolled runner style.

Each test file defines plain ``test_*`` functions and a ``TESTS`` list, then
``run(TESTS, title)`` executes them (AssertionError -> FAILED, other -> ERROR,
a returned "SKIP..." string -> skipped). The runner itself now lives in
``scanlation_sdk.testing`` (shared with the engine packages) and is re-exported
here so test modules keep importing it from ``tests.helpers``. The whole fast
suite runs via ``python -m tests`` from the core package.
"""
from __future__ import annotations

import base64
import hashlib
import io

from PIL import Image

from scanlation_sdk.testing import run, run_modules  # re-export; tests import these from here


def png_b64(width: int = 400, height: int = 300, color=(255, 255, 255)) -> str:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def md5_of(b64: str) -> str:
    """md5 over the base64 STRING — matches client content.js + server."""
    return hashlib.md5(b64.encode("utf-8")).hexdigest()


def payload(color=(255, 255, 255), width: int = 400, height: int = 300) -> dict:
    """An {b64, md5} image payload; a distinct color -> a distinct md5, so cache
    tests don't collide."""
    b64 = png_b64(width, height, color)
    return {"b64": b64, "md5": md5_of(b64)}


_CLIENT = None


def client():
    """Cached FastAPI TestClient — one app instance for the whole run (matches
    the old session-scoped fixture, so state persists across a file's tests)."""
    global _CLIENT
    if _CLIENT is None:
        from fastapi.testclient import TestClient

        from app.main import app
        from tests.fake_engines import install_fakes

        install_fakes()  # product ships no engine; register test fakes + select them
        _CLIENT = TestClient(app)
    return _CLIENT
