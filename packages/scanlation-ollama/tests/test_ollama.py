"""OllamaTranslator unit tests — request shape + parsing, HTTP mocked.

ollama itself runs on a separate (Linux/ROCm) host, so these never hit the
network: _post is replaced with a fake that captures the request body.
"""
from __future__ import annotations

import json

from scanlation_ollama.plugin import OllamaTranslator


def _translator() -> OllamaTranslator:
    """An OllamaTranslator whose _post is faked; tr._captured holds the body."""
    tr = OllamaTranslator()
    captured: dict = {}

    def fake_post(path, body):
        captured.clear()
        captured.update(body)
        return {"response": "  안녕하세요  ", "done": True}

    tr._post = fake_post
    tr._captured = captured
    return tr


def test_builds_request_from_tuned_config():
    translator = _translator()
    out = translator.translate("こんにちは", "ja", "ko", {"model": "gemma-test"})
    assert out == "안녕하세요"  # response stripped

    body = translator._captured
    assert body["model"] == "gemma-test"  # model comes from options (admin), no env fallback
    assert body["stream"] is False
    assert body["think"] is False
    assert body["system"].startswith("You are a translator")
    # plain language names + prompt template
    assert 'src="japanese"' in body["prompt"]
    assert 'dst="korean"' in body["prompt"]
    assert 'text="こんにちは"' in body["prompt"]
    # user's tuned options
    o = body["options"]
    assert o == {"temperature": 0.0, "seed": 42, "top_p": 1.0, "repeat_penalty": 1.1, "frequency_penalty": 0.0, "num_gpu": 31, "num_ctx": 2048}


def test_options_override():
    translator = _translator()
    translator.translate("テスト文章です", "ja", "ko", {"model": "gemma-test", "num_ctx": 1024, "think": True, "temperature": 0.7})
    body = translator._captured
    assert body["think"] is True
    assert body["options"]["num_ctx"] == 1024
    assert body["options"]["temperature"] == 0.7


def test_blank_skips_but_short_text_translates():
    tr = OllamaTranslator()
    calls = {"n": 0}

    def fake(path, body):
        calls["n"] += 1
        return {"response": "x"}

    tr._post = fake
    assert tr.translate("  ", "ja", "ko", {}) == ""              # blank -> no model call
    assert calls["n"] == 0
    assert tr.translate("あ", "ja", "ko", {"model": "m"}) == "x"  # 1-char now goes to the model
    assert calls["n"] == 1


def test_missing_model_raises():
    tr = OllamaTranslator()
    tr._post = lambda path, body: {"response": "x"}  # must never be reached
    raised = False
    try:
        tr.translate("これは十分に長い文章です", "ja", "ko", {})  # no model in options
    except ValueError:
        raised = True
    assert raised, "translate must raise when no model is selected"


def test_batch_builds_format_request_and_aligns():
    tr = OllamaTranslator()
    captured: dict = {}

    def fake_post(path, body):
        captured.clear()
        captured.update(body)
        return {"response": json.dumps({"t0": "가", "t1": "나"})}

    tr._post = fake_post
    out = tr.translate_batch(["日本語一", "日本語二"], "ja", "ko", {"model": "m"})
    assert out == ["가", "나"]                              # aligned to input order
    assert captured["format"]["required"] == ["t0", "t1"]  # schema forces exactly 2 keys
    assert captured["options"]["num_ctx"] == 2048          # single + batch share one num_ctx (no reload)
    assert 'src="japanese"' in captured["prompt"] and 'dst="korean"' in captured["prompt"]


def test_batch_passes_through_blanks():
    tr = OllamaTranslator()
    calls = {"n": 0}

    def fake_post(path, body):
        calls["n"] += 1
        return {"response": json.dumps({"t0": "가", "t1": "나"})}  # both non-blank texts batched

    tr._post = fake_post
    out = tr.translate_batch(["あ", "  ", "長い文章"], "ja", "ko", {"model": "m"})
    assert out == ["가", "", "나"]  # blank kept in place; short + long both translated, aligned
    assert calls["n"] == 1          # one batch call covers both non-blank texts


def test_batch_falls_back_to_per_text_on_bad_json():
    tr = OllamaTranslator()
    calls = {"n": 0}

    def fake_post(path, body):
        calls["n"] += 1
        if "format" in body:          # the batch attempt -> return unparseable garbage
            return {"response": "not json"}
        return {"response": "폴백"}   # per-text fallback calls

    tr._post = fake_post
    out = tr.translate_batch(["長い文章その一", "長い文章その二"], "ja", "ko", {"model": "m"})
    assert out == ["폴백", "폴백"]     # fallback filled both, aligned
    assert calls["n"] == 3            # 1 failed batch + 2 per-text


TESTS = [
    test_builds_request_from_tuned_config,
    test_options_override,
    test_blank_skips_but_short_text_translates,
    test_missing_model_raises,
    test_batch_builds_format_request_and_aligns,
    test_batch_passes_through_blanks,
    test_batch_falls_back_to_per_text_on_bad_json,
]

if __name__ == "__main__":
    import sys

    from scanlation_sdk.testing import run

    sys.exit(run(TESTS, "test_ollama"))
