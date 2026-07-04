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


def test_set_device_switches_and_validates():
    """Device is a persisted global (detector + recognizer); switching it drops
    cached engine instances so the next request reloads on the new device."""
    from scanlation_sdk.context import context
    from app.state import state

    c = client()
    saved = state.selection.device
    try:
        r = c.post("/set_device/", json={"device": "cuda"})
        assert r.status_code == 200 and r.json()["device"] == "cuda"
        # persisted + surfaced to the admin, and the shared context is updated
        assert c.get("/get_settings/").json()["selection"]["device"] == "cuda"
        assert context.device == "cuda"
        assert c.post("/set_device/", json={"device": "cpu"}).json()["device"] == "cpu"
        # unknown device -> 400
        assert c.post("/set_device/", json={"device": "tpu"}).status_code == 400
    finally:
        state.set_device(saved)


TESTS = [
    test_set_engines_validates,
    test_set_languages_validates,
    test_set_device_switches_and_validates,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_routes_settings"))
