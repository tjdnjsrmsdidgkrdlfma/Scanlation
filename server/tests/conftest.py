"""Shared fixtures. Sets an isolated data dir BEFORE app import so tests never
touch real cache/state files.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import tempfile

import pytest
from PIL import Image

# Must run before any `app.*` import so config.settings picks it up.
os.environ.setdefault("SCANLATION_BASE_DIR", tempfile.mkdtemp(prefix="scanlation-test-"))


def png_b64(width: int = 400, height: int = 300, color=(255, 255, 255)) -> str:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def md5_of(b64: str) -> str:
    """md5 over the base64 STRING — matches client content.js + server."""
    return hashlib.md5(b64.encode("utf-8")).hexdigest()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture
def image_payload():
    b64 = png_b64()
    return {"b64": b64, "md5": md5_of(b64)}


@pytest.fixture
def make_payload():
    """Factory: distinct color -> distinct md5, so cache tests don't collide."""
    def _make(color=(255, 255, 255), width=400, height=300):
        b64 = png_b64(width, height, color)
        return {"b64": b64, "md5": md5_of(b64)}

    return _make
