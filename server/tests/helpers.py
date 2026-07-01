"""Shared test helpers — no pytest, matching the hand-rolled runner style.

Each test file defines plain ``test_*`` functions and a ``TESTS`` list, then
``run(TESTS, title)`` executes them (AssertionError -> FAILED, other -> ERROR,
a returned "SKIP..." string -> skipped). The whole fast suite runs via
``python -m tests`` from ``server/``; a single file via ``python -m tests.<mod>``.
"""
from __future__ import annotations

import base64
import hashlib
import io

from PIL import Image


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

        _CLIENT = TestClient(app)
    return _CLIENT


def run(tests, title: str) -> int:
    """Run zero-arg test callables; print O/X/- per test; return 0 (all ok) or 1.
    A test returning a string starting with 'SKIP' is reported as skipped."""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    results: dict[str, str] = {}
    for test in tests:
        name = test.__name__
        try:
            r = test()
            results[name] = r if isinstance(r, str) and r.startswith("SKIP") else "PASSED"
        except AssertionError as e:
            results[name] = f"FAILED: {e}"
        except Exception as e:  # noqa: BLE001
            results[name] = f"ERROR: {type(e).__name__}: {e}"
    for name, res in results.items():
        status = "O" if res == "PASSED" else ("-" if res.startswith("SKIP") else "X")
        print(f"  {status} {name}: {res}")
    return 0 if all(r == "PASSED" or r.startswith("SKIP") for r in results.values()) else 1
