"""routes/settings_routes.py tests — engine/language/device selection endpoints."""
from __future__ import annotations

from tests.helpers import client, run


def test_set_engines_validates():
    c = client()
    assert c.post(
        "/set_engines/",
        json={"detector": "dummy", "recognizer": "dummy", "translator": "dummy"},
    ).status_code == 200
    assert c.post("/set_engines/", json={"detector": "nope"}).status_code == 400


def test_set_languages_validates():
    c = client()
    assert c.post("/set_languages/", json={"lang_src": "ja", "lang_dst": "ko"}).status_code == 200
    assert c.post("/set_languages/", json={"lang_src": "xx", "lang_dst": "ko"}).status_code == 400


def test_set_engine_device_validates():
    """Per-engine device override: set it, remove it (empty -> engine default),
    reject a bad device and an unknown engine. A real change drops that engine's
    cached instance so it reloads on the resolved device."""
    from app.state import state

    c = client()
    try:
        r = c.post("/set_engine_device/", json={"engine": "dummy", "device": "cuda"})
        assert r.status_code == 200 and r.json()["device"] == "cuda"
        assert state.resolve_device_for("dummy") == "cuda"
        # an indexed GPU is accepted (format-only validation) and persists verbatim
        r = c.post("/set_engine_device/", json={"engine": "dummy", "device": "cuda:1"})
        assert r.status_code == 200 and r.json()["device"] == "cuda:1"
        assert state.resolve_device_for("dummy") == "cuda:1"
        # blank removes the override -> back to the engine's DEFAULT_DEVICE
        assert c.post("/set_engine_device/", json={"engine": "dummy", "device": ""}).json()["device"] == ""
        assert state.resolve_device_for("dummy") is None
        # malformed / unknown devices -> 400
        assert c.post("/set_engine_device/", json={"engine": "dummy", "device": "tpu"}).status_code == 400
        assert c.post("/set_engine_device/", json={"engine": "dummy", "device": "cuda:x"}).status_code == 400
        assert c.post("/set_engine_device/", json={"engine": "dummy", "device": "gpu"}).status_code == 400
        # unknown engine -> 400
        assert c.post("/set_engine_device/", json={"engine": "nope", "device": "cpu"}).status_code == 400
    finally:
        state.set_engine_device("dummy", None)


TESTS = [
    test_set_engines_validates,
    test_set_languages_validates,
    test_set_engine_device_validates,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_routes_settings"))
