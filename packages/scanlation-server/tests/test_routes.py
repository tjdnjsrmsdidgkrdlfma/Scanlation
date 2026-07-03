"""Wire-protocol tests, all via dummy engines (zero model risk). Proves the
md5/box/lazy contract and the detector/recognizer/translator field names.
"""
from __future__ import annotations

from tests.helpers import client, payload, run


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


def test_run_pipeline_work_returns_boxes():
    p = payload()
    r = client().post("/run_pipeline/", json={"md5": p["md5"], "contents": p["b64"]})
    assert r.status_code == 200
    result = r.json()["result"]
    assert len(result) == 2
    for item in result:
        assert set(item) == {"ocr", "tsl", "box"}
        assert len(item["box"]) == 4


def test_run_pipeline_md5_mismatch_is_400():
    p = payload()
    r = client().post("/run_pipeline/", json={"md5": "deadbeef", "contents": p["b64"]})
    assert r.status_code == 400


def test_no_engine_installed_is_400():
    """The core ships no engine; running a role with none selected -> 400."""
    from app.state import state

    c = client()
    saved = (state.selection.detector, state.selection.recognizer, state.selection.translator)
    try:
        state.selection.detector = ""                 # no detector installed/selected
        p = payload(color=(7, 7, 7))                  # unique md5 -> cache miss -> runs
        r = c.post("/run_pipeline/", json={"md5": p["md5"], "contents": p["b64"]})
        assert r.status_code == 400
    finally:
        state.selection.detector, state.selection.recognizer, state.selection.translator = saved


def test_lazy_miss_then_work_then_cached_hit():
    c = client()
    p = payload(color=(123, 222, 31))  # unique md5

    # lazy with unknown md5 -> non-2xx so the client falls through to work
    miss = c.post("/run_pipeline/", json={"md5": p["md5"]})
    assert miss.status_code >= 400

    # work populates the cache
    work = c.post("/run_pipeline/", json={"md5": p["md5"], "contents": p["b64"]})
    assert work.status_code == 200

    # lazy again -> served from cache
    hit = c.post("/run_pipeline/", json={"md5": p["md5"]})
    assert hit.status_code == 200
    assert hit.json()["result"] == work.json()["result"]


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


def test_catalog_lists_engines():
    from app.plugins_install import catalog

    c = catalog()
    for name in ("rtdetr", "mangaocr", "ollama", "llamacpp"):
        assert name in c, name
    assert "detector" in c["rtdetr"].roles
    assert "recognizer" in c["mangaocr"].roles
    assert "translator" in c["ollama"].roles
    assert c["rtdetr"].package == "scanlation-rtdetr"


def test_install_package_builds_pip_git_command():
    """Default install shells out to `pip install --target=<vol> <sdk git+> <engine
    git+>` — no engine code is baked in; it's fetched from the repo. (Verified
    without actually installing.)"""
    import os

    from app import plugins_install as pi

    entry = pi.catalog()["ollama"]
    recorded = {}

    class _Ok:
        returncode = 0
        stderr = ""
        stdout = ""

    orig_run = pi.subprocess.run
    orig_src = os.environ.pop("SCANLATION_ENGINES_SRC", None)  # force git mode
    pi.subprocess.run = lambda cmd, **kw: (recorded.__setitem__("cmd", cmd), _Ok())[1]
    try:
        pi.install_package(entry)
    finally:
        pi.subprocess.run = orig_run
        if orig_src is not None:
            os.environ["SCANLATION_ENGINES_SRC"] = orig_src

    cmd = recorded["cmd"]
    assert cmd[1:5] == ["-m", "pip", "install", "--target"]
    assert str(pi.plugins_dir()) in cmd
    joined = " ".join(cmd)
    assert "git+" in joined
    assert "#subdirectory=packages/scanlation-ollama" in joined  # the engine
    assert "#subdirectory=packages/scanlation-sdk" in joined     # co-installed sdk


def test_install_plugins():
    c = client()
    # dummy has no assets -> install is a no-op success
    r = c.post("/install_plugins/", json={"plugins": {"dummy": True}})
    assert r.status_code == 200 and r.json()["status"] == "success"
    # unknown plugin -> 502
    r2 = c.post("/install_plugins/", json={"plugins": {"nope": True}})
    assert r2.status_code == 502


# --- admin: get_settings / models / set_options / prompts ------------------
def test_get_settings_shape():
    d = client().get("/get_settings/").json()
    assert set(d) >= {"version", "selection", "languages", "engines", "prompts"}
    assert set(d["engines"]) == {"detector", "recognizer", "translator"}
    assert "default" in d["prompts"]["builtin"]            # default always present
    # engines carry their OPTION_SCHEMA so the admin can render option fields
    det = {e["name"]: e for e in d["engines"]["detector"]}
    assert "dummy" in det and "num_boxes" in det["dummy"]["schema"]
    assert det["dummy"]["schema"]["num_boxes"]["type"] == "int"


def test_get_settings_merges_catalog():
    """Installable-but-not-installed engines are merged in from the catalog so the
    admin can install them; every entry carries the installed_package flag."""
    d = client().get("/get_settings/").json()
    names = set()
    for role in ("detector", "recognizer", "translator"):
        for e in d["engines"][role]:
            names.add(e["name"])
            assert "installed_package" in e
    for name in ("rtdetr", "mangaocr", "ollama", "llamacpp"):
        assert name in names, name


def test_get_translator_models_shape():
    c = client()
    # active translator is dummy (no backend) -> empty list, never errors
    d = c.get("/get_translator_models/").json()
    assert isinstance(d["models"], list) and d["models"] == []
    # unknown engine -> empty, not a 4xx
    assert c.get("/get_translator_models/", params={"engine": "nope"}).json()["models"] == []


def test_set_options_persists_and_clears():
    c = client()
    r = c.post("/set_options/", json={"engine": "dummy", "options": {"num_boxes": 1}})
    assert r.status_code == 200
    det = {e["name"]: e for e in c.get("/get_settings/").json()["engines"]["detector"]}
    assert det["dummy"]["options"]["num_boxes"] == 1
    # blank value removes that override (reverts to the schema default)
    c.post("/set_options/", json={"engine": "dummy", "options": {"num_boxes": ""}})
    det = {e["name"]: e for e in c.get("/get_settings/").json()["engines"]["detector"]}
    assert "num_boxes" not in det["dummy"]["options"]
    # unknown engine -> 400
    assert c.post("/set_options/", json={"engine": "nope", "options": {}}).status_code == 400


def test_prompt_select_save_delete():
    c = client()
    assert c.post("/select_prompt/", json={"name": "literal"}).json()["active"] == "literal"
    assert c.post("/select_prompt/", json={"name": "ghost"}).status_code == 400  # unknown
    # save custom -> active + listed under custom
    c.post("/save_prompt/", json={"name": "mine", "text": "SYSTEM TEST PROMPT"})
    p = c.get("/get_settings/").json()["prompts"]
    assert p["active"] == "mine" and p["custom"]["mine"] == "SYSTEM TEST PROMPT"
    assert c.post("/delete_prompt/", json={"name": "default"}).status_code == 400  # builtin protected
    # delete custom -> active falls back to default
    assert c.post("/delete_prompt/", json={"name": "mine"}).json()["active"] == "default"
    c.post("/select_prompt/", json={"name": "default"})  # cleanup


def test_active_prompt_injected_into_translator_options():
    from app.state import state

    c = client()
    c.post("/save_prompt/", json={"name": "inj", "text": "INJECTED-PROMPT"})
    assert state.translator_options("dummy", None)["system_prompt"] == "INJECTED-PROMPT"
    c.post("/delete_prompt/", json={"name": "inj"})  # cleanup -> back to default
    assert state.translator_options("dummy", None)["system_prompt"].startswith("From now on")


def test_clear_cache_drops_runs_and_translations():
    from app.cache import cache

    c = client()
    p = payload(color=(7, 9, 11))  # unique md5
    # populate both caches: a page result (ocr_runs) + translation-log rows.
    # run_ocrtsl records each recognized text -> its translation in the TM; the
    # dummy recognizer emits "REGION-<order>", so "REGION-0" lands in the log.
    assert c.post("/run_pipeline/", json={"md5": p["md5"], "contents": p["b64"]}).status_code == 200
    assert c.post("/run_pipeline/", json={"md5": p["md5"]}).status_code == 200  # lazy hit = cached
    assert cache.get_translations("REGION-0", "ja", "ko")  # TM non-empty

    r = c.post("/clear_cache/", json={})
    assert r.status_code == 200
    assert r.json()["status"] == "success" and r.json()["cleared"] >= 1

    # page cache gone -> lazy now misses (client would fall through to work)
    assert c.post("/run_pipeline/", json={"md5": p["md5"]}).status_code == 404
    # translation log gone too
    assert cache.get_translations("REGION-0", "ja", "ko") == []


def test_client_config_min_image_dim():
    c = client()
    # handshake + get_settings expose the current value
    assert "min_image_dim" in c.get("/").json()
    assert "min_image_dim" in c.get("/get_settings/").json()["selection"]
    try:
        # set it -> reflected in both the handshake and the admin snapshot
        r = c.post("/set_client_config/", json={"min_image_dim": 120})
        assert r.status_code == 200 and r.json()["min_image_dim"] == 120
        assert c.get("/").json()["min_image_dim"] == 120
        assert c.get("/get_settings/").json()["selection"]["min_image_dim"] == 120
        # negative -> 400
        assert c.post("/set_client_config/", json={"min_image_dim": -5}).status_code == 400
    finally:
        c.post("/set_client_config/", json={"min_image_dim": 80})  # cleanup


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
    test_run_pipeline_work_returns_boxes,
    test_run_pipeline_md5_mismatch_is_400,
    test_no_engine_installed_is_400,
    test_lazy_miss_then_work_then_cached_hit,
    test_set_engines_validates,
    test_set_languages_validates,
    test_set_device_switches_and_validates,
    test_catalog_lists_engines,
    test_install_package_builds_pip_git_command,
    test_install_plugins,
    test_get_settings_shape,
    test_get_settings_merges_catalog,
    test_get_translator_models_shape,
    test_set_options_persists_and_clears,
    test_prompt_select_save_delete,
    test_active_prompt_injected_into_translator_options,
    test_clear_cache_drops_runs_and_translations,
    test_client_config_min_image_dim,
    test_auth_token_gates_when_set,
    test_auth_preflight_open_and_401_carries_cors,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_routes"))
