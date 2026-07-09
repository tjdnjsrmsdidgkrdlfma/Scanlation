"""routes/admin.py tests — get_settings / models / set_options / prompts /
cache clearing / client config."""
from __future__ import annotations

from tests.helpers import client, payload, run


def test_get_settings_shape():
    d = client().get("/get_settings/").json()
    assert set(d) >= {"version", "selection", "languages", "engines", "prompts"}
    assert "gpus" in d and isinstance(d["gpus"], list)      # host GPU inventory ([] on CPU-only)
    assert d["selection"]["torch_backend"] in ("cpu", "gpu")  # GPU/torch backend for installs
    assert "gpu_vendor" in d                                # amd/nvidia/both/None (device nodes)
    assert "torch_build" in d                               # cpu/cuda/rocm/None (installed torch)
    assert set(d["engines"]) == {"detector", "recognizer", "translator"}
    assert "default" in d["prompts"]["builtin"]            # default always present
    # engines carry their OPTION_SCHEMA so the admin can render option fields
    det = {e["name"]: e for e in d["engines"]["detector"]}
    assert "dummy" in det and "num_boxes" in det["dummy"]["schema"]
    assert det["dummy"]["schema"]["num_boxes"]["type"] == "int"
    # per-engine device fields: the fake is EngineBase-only, so it doesn't use a device
    assert det["dummy"]["uses_device"] is False
    assert "device" in det["dummy"] and det["dummy"]["default_device"] == "cpu"


def test_get_settings_merges_catalog():
    """Installable-but-not-installed engines are merged in from the catalog so the
    admin can install them; every entry carries the installed_package flag."""
    d = client().get("/get_settings/").json()
    names = set()
    for role in ("detector", "recognizer", "translator"):
        for e in d["engines"][role]:
            names.add(e["name"])
            assert "installed_package" in e
    for name in ("comic-text-and-bubble-detector", "manga-ocr", "Ollama", "llama.cpp"):
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
    assert c.post("/select_prompt/", json={"name": "default"}).json()["active"] == "default"
    assert c.post("/select_prompt/", json={"name": "ghost"}).status_code == 400  # unknown
    # save custom -> active + listed under custom
    c.post("/save_prompt/", json={"name": "mine", "text": "SYSTEM TEST PROMPT"})
    p = c.get("/get_settings/").json()["prompts"]
    assert p["active"] == "mine" and p["custom"]["mine"] == "SYSTEM TEST PROMPT"
    # select an existing custom by name
    assert c.post("/select_prompt/", json={"name": "mine"}).json()["active"] == "mine"
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
    assert state.translator_options("dummy", None)["system_prompt"].startswith("You are a translator")


def test_clear_cache_drops_runs():
    c = client()
    p = payload(color=(7, 9, 11))  # unique md5
    # populate the page-result cache (page_runs)
    assert c.post("/run_pipeline/", json={"md5": p["md5"], "contents": p["b64"]}).status_code == 200
    assert c.post("/run_lookup/", json={"md5": p["md5"]}).json()["result"] is not None  # cached

    r = c.post("/clear_cache/", json={})
    assert r.status_code == 200
    assert r.json()["status"] == "success" and r.json()["cleared"] >= 1

    # page cache gone -> lookup now misses (200 {result: null}; client falls through to work)
    assert c.post("/run_lookup/", json={"md5": p["md5"]}).json()["result"] is None


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
        # negative -> clamped to 0 (state is the single validation authority, not a 400)
        r = c.post("/set_client_config/", json={"min_image_dim": -5})
        assert r.status_code == 200 and r.json()["min_image_dim"] == 0
    finally:
        c.post("/set_client_config/", json={"min_image_dim": 80})  # cleanup


def test_client_config_verbose_log():
    c = client()
    # server-only (NOT in the handshake the extension reads); shown in the admin snapshot
    assert "verbose_log" not in c.get("/").json()
    assert "verbose_log" in c.get("/get_settings/").json()["selection"]
    try:
        r = c.post("/set_client_config/", json={"verbose_log": True})
        assert r.status_code == 200 and r.json()["verbose_log"] is True
        assert c.get("/get_settings/").json()["selection"]["verbose_log"] is True
    finally:
        c.post("/set_client_config/", json={"verbose_log": False})  # cleanup (also resets the live logger)


def test_client_config_translate_concurrency():
    c = client()
    # server-only (NOT in the handshake the extension reads); shown in the admin snapshot
    assert "translate_concurrency" not in c.get("/").json()
    assert "translate_concurrency" in c.get("/get_settings/").json()["selection"]
    try:
        r = c.post("/set_client_config/", json={"translate_concurrency": 8})
        assert r.status_code == 200 and r.json()["translate_concurrency"] == 8
        assert c.get("/get_settings/").json()["selection"]["translate_concurrency"] == 8
        # below 1 -> clamped to 1 (state is the single validation authority, not a 400)
        r = c.post("/set_client_config/", json={"translate_concurrency": 0})
        assert r.status_code == 200 and r.json()["translate_concurrency"] == 1
    finally:
        c.post("/set_client_config/", json={"translate_concurrency": 1})  # cleanup (back to default)


def test_client_config_model_idle_unload_minutes():
    c = client()
    # server-only (NOT in the handshake the extension reads); shown in the admin snapshot
    assert "model_idle_unload_minutes" not in c.get("/").json()
    assert "model_idle_unload_minutes" in c.get("/get_settings/").json()["selection"]
    try:
        r = c.post("/set_client_config/", json={"model_idle_unload_minutes": 15})
        assert r.status_code == 200 and r.json()["model_idle_unload_minutes"] == 15
        assert c.get("/get_settings/").json()["selection"]["model_idle_unload_minutes"] == 15
        # 0 stays 0 (disabled — keep resident); negative clamps to 0 (state is the
        # single validation authority, not a 400)
        r = c.post("/set_client_config/", json={"model_idle_unload_minutes": 0})
        assert r.status_code == 200 and r.json()["model_idle_unload_minutes"] == 0
        r = c.post("/set_client_config/", json={"model_idle_unload_minutes": -3})
        assert r.status_code == 200 and r.json()["model_idle_unload_minutes"] == 0
    finally:
        c.post("/set_client_config/", json={"model_idle_unload_minutes": 5})  # cleanup (back to default)


TESTS = [
    test_get_settings_shape,
    test_get_settings_merges_catalog,
    test_get_translator_models_shape,
    test_set_options_persists_and_clears,
    test_prompt_select_save_delete,
    test_active_prompt_injected_into_translator_options,
    test_clear_cache_drops_runs,
    test_client_config_min_image_dim,
    test_client_config_verbose_log,
    test_client_config_translate_concurrency,
    test_client_config_model_idle_unload_minutes,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_routes_admin"))
