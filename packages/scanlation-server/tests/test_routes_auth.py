"""routes/handshake.py + main.py middleware tests — handshake shape and the
X-Auth-Token gate (with CORS interplay)."""
from __future__ import annotations

from tests.helpers import client, run


def test_handshake_keys():
    r = client().get("/")
    assert r.status_code == 200
    d = r.json()
    for key in (
        "version", "Languages", "Languages_src", "Languages_dst", "Languages_hr",
        "detectors", "recognizers", "translators",
        "detector_selected", "recognizer_selected", "translator_selected", "lang_src", "lang_dst",
    ):
        assert key in d, key
    assert "dummy" in d["detectors"]
    assert len(d["Languages"]) == len(d["Languages_hr"])
    assert d["lang_src"] == "ja" and d["lang_dst"] == "ko"


def test_auth_token_gates_when_set():
    from app.config import settings
    c = client()
    settings.auth_token = "s3cret"
    try:
        assert c.get("/").status_code == 401                                  # no token
        assert c.get("/", headers={"X-Auth-Token": "nope"}).status_code == 401  # wrong token
        assert c.get("/", headers={"X-Auth-Token": "s3cret"}).status_code == 200  # right token
        # a mutating endpoint is gated too
        assert c.post("/set_languages/", json={"lang_src": "ja", "lang_dst": "ko"}).status_code == 401
        # the /admin static shell stays open so the token can be entered
        assert c.get("/admin/").status_code == 200
    finally:
        settings.auth_token = ""
    assert c.get("/").status_code == 200  # cleared -> open again (current default)


def test_auth_preflight_open_and_401_carries_cors():
    from app.config import settings
    c = client()
    settings.auth_token = "s3cret"
    try:
        # CORS preflight (no token) must pass, else the extension's cross-origin calls die
        pre = c.options("/run_pipeline/", headers={
            "Origin": "https://example.com", "Access-Control-Request-Method": "POST",
        })
        assert pre.status_code < 400
        # the 401 still carries CORS headers (CORS middleware is outermost)
        r = c.get("/", headers={"Origin": "https://example.com"})
        assert r.status_code == 401
        assert r.headers.get("access-control-allow-origin") == "*"
    finally:
        settings.auth_token = ""


TESTS = [
    test_handshake_keys,
    test_auth_token_gates_when_set,
    test_auth_preflight_open_and_401_carries_cors,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_routes_auth"))
