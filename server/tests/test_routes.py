"""Wire-protocol tests, all via dummy engines (zero model risk). Proves the
md5/box/lazy contract and the detector/recognizer/translator field names.
"""
from __future__ import annotations


def test_handshake_keys(client):
    r = client.get("/")
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


def test_run_ocrtsl_work_returns_boxes(client, image_payload):
    r = client.post("/run_ocrtsl/", json={"md5": image_payload["md5"], "contents": image_payload["b64"]})
    assert r.status_code == 200
    result = r.json()["result"]
    assert len(result) == 2
    for item in result:
        assert set(item) == {"ocr", "tsl", "box"}
        assert len(item["box"]) == 4


def test_run_ocrtsl_md5_mismatch_is_400(client, image_payload):
    r = client.post("/run_ocrtsl/", json={"md5": "deadbeef", "contents": image_payload["b64"]})
    assert r.status_code == 400


def test_lazy_miss_then_work_then_cached_hit(client, make_payload):
    p = make_payload(color=(123, 222, 31))  # unique md5

    # lazy with unknown md5 -> non-2xx so the client falls through to work
    miss = client.post("/run_ocrtsl/", json={"md5": p["md5"]})
    assert miss.status_code >= 400

    # work populates the cache
    work = client.post("/run_ocrtsl/", json={"md5": p["md5"], "contents": p["b64"]})
    assert work.status_code == 200

    # lazy again -> served from cache
    hit = client.post("/run_ocrtsl/", json={"md5": p["md5"]})
    assert hit.status_code == 200
    assert hit.json()["result"] == work.json()["result"]


def test_run_tsl_and_get_trans_roundtrip(client):
    text = "こんにちは"
    r = client.post("/run_tsl/", json={"text": text})
    assert r.status_code == 200
    assert r.json()["text"] == f"[ja->ko] {text}"

    g = client.get("/get_trans/", params={"text": text})
    assert g.status_code == 200
    models = {t["model"] for t in g.json()["translations"]}
    assert "dummy" in models


def test_set_manual_translation_wins(client):
    text = "手動テスト"
    client.post("/set_manual_translation/", json={"text": text, "translation": "수동번역"})

    g = client.get("/get_trans/", params={"text": text}).json()["translations"]
    assert any(t["model"] == "manual" and t["text"] == "수동번역" for t in g)

    # run_tsl must honor the manual override
    r = client.post("/run_tsl/", json={"text": text})
    assert r.json()["text"] == "수동번역"


def test_get_active_options_shape(client):
    d = client.get("/get_active_options/").json()["options"]
    assert set(d) == {"detector", "recognizer", "translator"}
    assert d["detector"]["num_boxes"]["type"] == "int"
    assert d["detector"]["num_boxes"]["default"] == 2


def test_set_models_validates(client):
    assert client.post(
        "/set_models/",
        json={"detector": "dummy", "recognizer": "dummy", "translator": "dummy"},
    ).status_code == 200
    assert client.post("/set_models/", json={"detector": "nope"}).status_code == 400


def test_set_lang_validates(client):
    assert client.post("/set_lang/", json={"lang_src": "ja", "lang_dst": "ko"}).status_code == 200
    assert client.post("/set_lang/", json={"lang_src": "xx", "lang_dst": "ko"}).status_code == 400


def test_get_plugin_data_lists_engines(client):
    d = client.get("/get_plugin_data/").json()
    assert "dummy" in d and "ctd" in d
    assert d["dummy"]["installed"] is True            # no downloadable assets
    assert isinstance(d["ctd"]["installed"], bool)    # real status (depends on weights)


def test_manage_plugins_install(client):
    # dummy has no assets -> install is a no-op success
    r = client.post("/manage_plugins/", json={"plugins": {"dummy": True}})
    assert r.status_code == 200 and r.json()["status"] == "success"
    # unknown plugin -> 502
    r2 = client.post("/manage_plugins/", json={"plugins": {"nope": True}})
    assert r2.status_code == 502


# --- admin: get_settings / set_options / prompts ---------------------------
def test_get_settings_shape(client):
    d = client.get("/get_settings/").json()
    assert set(d) >= {"version", "selection", "languages", "engines", "prompts"}
    assert set(d["engines"]) == {"detector", "recognizer", "translator"}
    assert "default" in d["prompts"]["builtin"]            # default always present
    # the ollama translator exposes a 'model' option so the admin can set the tag
    tr = {e["name"]: e for e in d["engines"]["translator"]}
    assert "ollama" in tr and "model" in tr["ollama"]["schema"]
    assert tr["ollama"]["schema"]["model"]["type"] == "str"


def test_set_options_persists_and_clears(client):
    r = client.post("/set_options/", json={"engine": "ollama", "options": {"model": "gemma-x", "num_ctx": 1024}})
    assert r.status_code == 200
    tr = {e["name"]: e for e in client.get("/get_settings/").json()["engines"]["translator"]}
    assert tr["ollama"]["options"]["model"] == "gemma-x"
    assert tr["ollama"]["options"]["num_ctx"] == 1024
    # blank value removes that one override (reverts to schema/env default)
    client.post("/set_options/", json={"engine": "ollama", "options": {"model": ""}})
    tr = {e["name"]: e for e in client.get("/get_settings/").json()["engines"]["translator"]}
    assert "model" not in tr["ollama"]["options"] and tr["ollama"]["options"]["num_ctx"] == 1024
    client.post("/set_options/", json={"engine": "ollama", "options": {"num_ctx": ""}})  # cleanup
    # unknown engine -> 400
    assert client.post("/set_options/", json={"engine": "nope", "options": {}}).status_code == 400


def test_prompt_select_save_delete(client):
    assert client.post("/select_prompt/", json={"name": "literal"}).json()["active"] == "literal"
    assert client.post("/select_prompt/", json={"name": "ghost"}).status_code == 400  # unknown
    # save custom -> active + listed under custom
    client.post("/save_prompt/", json={"name": "mine", "text": "SYSTEM TEST PROMPT"})
    p = client.get("/get_settings/").json()["prompts"]
    assert p["active"] == "mine" and p["custom"]["mine"] == "SYSTEM TEST PROMPT"
    assert client.post("/delete_prompt/", json={"name": "default"}).status_code == 400  # builtin protected
    # delete custom -> active falls back to default
    assert client.post("/delete_prompt/", json={"name": "mine"}).json()["active"] == "default"
    client.post("/select_prompt/", json={"name": "default"})  # cleanup


def test_active_prompt_injected_into_translator_options(client):
    from app.state import state

    client.post("/save_prompt/", json={"name": "inj", "text": "INJECTED-PROMPT"})
    assert state.translator_options("ollama", None)["system_prompt"] == "INJECTED-PROMPT"
    client.post("/delete_prompt/", json={"name": "inj"})  # cleanup -> back to default
    assert state.translator_options("ollama", None)["system_prompt"].startswith("From now on")
