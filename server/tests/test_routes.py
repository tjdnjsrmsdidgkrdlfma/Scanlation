"""Wire-protocol compatibility with the ocr_extension client, all via dummy
engines (zero model risk). This is the test that proves md5/box/lazy are right.
"""
from __future__ import annotations


def test_handshake_keys(client):
    r = client.get("/")
    assert r.status_code == 200
    d = r.json()
    for key in (
        "version", "Languages", "Languages_src", "Languages_dst", "Languages_hr",
        "BOXModels", "OCRModels", "TSLModels",
        "box_selected", "ocr_selected", "tsl_selected", "lang_src", "lang_dst",
    ):
        assert key in d, key
    assert "dummy" in d["BOXModels"]
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
    assert set(d) == {"box_model", "ocr_model", "tsl_model"}
    assert d["box_model"]["num_boxes"]["type"] == "int"
    assert d["box_model"]["num_boxes"]["default"] == 2


def test_set_models_validates(client):
    assert client.post(
        "/set_models/",
        json={"box_model_id": "dummy", "ocr_model_id": "dummy", "tsl_model_id": "dummy"},
    ).status_code == 200
    assert client.post("/set_models/", json={"box_model_id": "nope"}).status_code == 400


def test_set_lang_validates(client):
    assert client.post("/set_lang/", json={"lang_src": "ja", "lang_dst": "ko"}).status_code == 200
    assert client.post("/set_lang/", json={"lang_src": "xx", "lang_dst": "ko"}).status_code == 400


def test_get_plugin_data_lists_engines(client):
    d = client.get("/get_plugin_data/").json()
    assert "dummy" in d and "ctd" in d
    assert d["ctd"]["installed"] is True


def test_manage_plugins_stub(client):
    r = client.post("/manage_plugins/", json={"plugins": {"dummy": True}})
    assert r.status_code == 200 and r.json()["status"] == "success"
