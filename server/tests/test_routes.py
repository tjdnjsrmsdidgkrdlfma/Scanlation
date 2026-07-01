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


def test_run_ocrtsl_work_returns_boxes():
    p = payload()
    r = client().post("/run_ocrtsl/", json={"md5": p["md5"], "contents": p["b64"]})
    assert r.status_code == 200
    result = r.json()["result"]
    assert len(result) == 2
    for item in result:
        assert set(item) == {"ocr", "tsl", "box"}
        assert len(item["box"]) == 4


def test_run_ocrtsl_md5_mismatch_is_400():
    p = payload()
    r = client().post("/run_ocrtsl/", json={"md5": "deadbeef", "contents": p["b64"]})
    assert r.status_code == 400


def test_lazy_miss_then_work_then_cached_hit():
    c = client()
    p = payload(color=(123, 222, 31))  # unique md5

    # lazy with unknown md5 -> non-2xx so the client falls through to work
    miss = c.post("/run_ocrtsl/", json={"md5": p["md5"]})
    assert miss.status_code >= 400

    # work populates the cache
    work = c.post("/run_ocrtsl/", json={"md5": p["md5"], "contents": p["b64"]})
    assert work.status_code == 200

    # lazy again -> served from cache
    hit = c.post("/run_ocrtsl/", json={"md5": p["md5"]})
    assert hit.status_code == 200
    assert hit.json()["result"] == work.json()["result"]


def test_run_tsl_and_get_trans_roundtrip():
    c = client()
    text = "こんにちは"
    r = c.post("/run_tsl/", json={"text": text})
    assert r.status_code == 200
    assert r.json()["text"] == f"[ja->ko] {text}"

    g = c.get("/get_trans/", params={"text": text})
    assert g.status_code == 200
    models = {t["model"] for t in g.json()["translations"]}
    assert "dummy" in models


def test_set_manual_translation_wins():
    c = client()
    text = "手動テスト"
    c.post("/set_manual_translation/", json={"text": text, "translation": "수동번역"})

    g = c.get("/get_trans/", params={"text": text}).json()["translations"]
    assert any(t["model"] == "manual" and t["text"] == "수동번역" for t in g)

    # run_tsl must honor the manual override
    r = c.post("/run_tsl/", json={"text": text})
    assert r.json()["text"] == "수동번역"


def test_get_active_options_shape():
    d = client().get("/get_active_options/").json()["options"]
    assert set(d) == {"detector", "recognizer", "translator"}
    assert d["detector"]["num_boxes"]["type"] == "int"
    assert d["detector"]["num_boxes"]["default"] == 2


def test_set_models_validates():
    c = client()
    assert c.post(
        "/set_models/",
        json={"detector": "dummy", "recognizer": "dummy", "translator": "dummy"},
    ).status_code == 200
    assert c.post("/set_models/", json={"detector": "nope"}).status_code == 400


def test_set_lang_validates():
    c = client()
    assert c.post("/set_lang/", json={"lang_src": "ja", "lang_dst": "ko"}).status_code == 200
    assert c.post("/set_lang/", json={"lang_src": "xx", "lang_dst": "ko"}).status_code == 400


def test_get_plugin_data_lists_engines():
    d = client().get("/get_plugin_data/").json()
    assert "dummy" in d and "ctd" in d
    assert d["dummy"]["installed"] is True            # no downloadable assets
    assert isinstance(d["ctd"]["installed"], bool)    # real status (depends on weights)


def test_manage_plugins_install():
    c = client()
    # dummy has no assets -> install is a no-op success
    r = c.post("/manage_plugins/", json={"plugins": {"dummy": True}})
    assert r.status_code == 200 and r.json()["status"] == "success"
    # unknown plugin -> 502
    r2 = c.post("/manage_plugins/", json={"plugins": {"nope": True}})
    assert r2.status_code == 502


# --- admin: get_settings / models / set_options / prompts ------------------
def test_get_settings_shape():
    d = client().get("/get_settings/").json()
    assert set(d) >= {"version", "selection", "languages", "engines", "prompts"}
    assert set(d["engines"]) == {"detector", "recognizer", "translator"}
    assert "default" in d["prompts"]["builtin"]            # default always present
    # the ollama translator exposes a 'model' option so the admin can set the tag
    tr = {e["name"]: e for e in d["engines"]["translator"]}
    assert "ollama" in tr and "model" in tr["ollama"]["schema"]
    assert tr["ollama"]["schema"]["model"]["type"] == "str"


def test_get_translator_models_shape():
    c = client()
    # active translator is dummy (no backend) -> empty list, never errors
    d = c.get("/get_translator_models/").json()
    assert isinstance(d["models"], list) and d["models"] == []
    # unknown engine -> empty, not a 4xx
    assert c.get("/get_translator_models/", params={"engine": "nope"}).json()["models"] == []


def test_set_options_persists_and_clears():
    c = client()
    r = c.post("/set_options/", json={"engine": "ollama", "options": {"model": "gemma-x", "num_ctx": 1024}})
    assert r.status_code == 200
    tr = {e["name"]: e for e in c.get("/get_settings/").json()["engines"]["translator"]}
    assert tr["ollama"]["options"]["model"] == "gemma-x"
    assert tr["ollama"]["options"]["num_ctx"] == 1024
    # blank value removes that one override (reverts to schema/env default)
    c.post("/set_options/", json={"engine": "ollama", "options": {"model": ""}})
    tr = {e["name"]: e for e in c.get("/get_settings/").json()["engines"]["translator"]}
    assert "model" not in tr["ollama"]["options"] and tr["ollama"]["options"]["num_ctx"] == 1024
    c.post("/set_options/", json={"engine": "ollama", "options": {"num_ctx": ""}})  # cleanup
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
    assert state.translator_options("ollama", None)["system_prompt"] == "INJECTED-PROMPT"
    c.post("/delete_prompt/", json={"name": "inj"})  # cleanup -> back to default
    assert state.translator_options("ollama", None)["system_prompt"].startswith("From now on")


TESTS = [
    test_handshake_keys,
    test_run_ocrtsl_work_returns_boxes,
    test_run_ocrtsl_md5_mismatch_is_400,
    test_lazy_miss_then_work_then_cached_hit,
    test_run_tsl_and_get_trans_roundtrip,
    test_set_manual_translation_wins,
    test_get_active_options_shape,
    test_set_models_validates,
    test_set_lang_validates,
    test_get_plugin_data_lists_engines,
    test_manage_plugins_install,
    test_get_settings_shape,
    test_get_translator_models_shape,
    test_set_options_persists_and_clears,
    test_prompt_select_save_delete,
    test_active_prompt_injected_into_translator_options,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_routes"))
