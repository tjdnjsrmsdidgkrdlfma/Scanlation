"""OllamaTranslator unit tests — request shape + parsing, HTTP mocked.

ollama itself runs on a separate (Linux/ROCm) host, so these never hit the
network: _generate is replaced with a fake that captures the request body.
"""
from __future__ import annotations

import json

from scanlation_ollama.plugin import OllamaTranslator


def _translator() -> OllamaTranslator:
    """An OllamaTranslator whose _generate is faked; tr._captured holds the body."""
    tr = OllamaTranslator()
    captured: dict = {}

    def fake_generate(body):
        captured.clear()
        captured.update(body)
        return {"response": "  안녕하세요  ", "done": True}

    tr._generate = fake_generate
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
    assert body["system"].startswith("From now on")
    # plain language names + prompt template
    assert 'src="japanese"' in body["prompt"]
    assert 'dst="korean"' in body["prompt"]
    assert 'text="こんにちは"' in body["prompt"]
    # user's tuned options
    o = body["options"]
    assert o == {"temperature": 0.0, "seed": 42, "top_p": 1.0, "num_gpu": 31, "num_ctx": 512}


def test_options_override():
    translator = _translator()
    translator.translate("テスト文章です", "ja", "ko", {"model": "gemma-test", "num_ctx": 1024, "think": True, "temperature": 0.7})
    body = translator._captured
    assert body["think"] is True
    assert body["options"]["num_ctx"] == 1024
    assert body["options"]["temperature"] == 0.7


def test_short_text_skips_model_call():
    tr = OllamaTranslator()
    called = False

    def fake(body):
        nonlocal called
        called = True
        return {"response": "x"}

    tr._generate = fake
    assert tr.translate("あ", "ja", "ko", {}) == "あ"  # <=2 chars returned as-is
    assert tr.translate("  ", "ja", "ko", {}) == ""
    assert called is False


def test_missing_model_raises():
    tr = OllamaTranslator()
    tr._generate = lambda body: {"response": "x"}  # must never be reached
    raised = False
    try:
        tr.translate("これは十分に長い文章です", "ja", "ko", {})  # no model in options
    except ValueError:
        raised = True
    assert raised, "translate must raise when no model is selected"


def test_batch_builds_format_request_and_aligns():
    tr = OllamaTranslator()
    captured: dict = {}

    def fake_generate(body):
        captured.clear()
        captured.update(body)
        return {"response": json.dumps({"t0": "가", "t1": "나"})}

    tr._generate = fake_generate
    out = tr.translate_batch(["日本語一", "日本語二"], "ja", "ko", {"model": "m"})
    assert out == ["가", "나"]                              # aligned to input order
    assert captured["format"]["required"] == ["t0", "t1"]  # schema forces exactly 2 keys
    assert captured["options"]["num_ctx"] == 2048          # batch bumps ctx (single stays 512)
    assert 'src="japanese"' in captured["prompt"] and 'dst="korean"' in captured["prompt"]


def test_batch_passes_through_short_texts():
    tr = OllamaTranslator()
    calls = {"n": 0}

    def fake_generate(body):
        calls["n"] += 1
        return {"response": json.dumps({"t0": "번역"})}  # only the one long text is batched

    tr._generate = fake_generate
    out = tr.translate_batch(["あ", "これは十分に長い文章", "  "], "ja", "ko", {"model": "m"})
    assert out == ["あ", "번역", ""]  # short/empty kept in place, long one translated
    assert calls["n"] == 1            # exactly one model call (for the single long text)


def test_batch_falls_back_to_per_text_on_bad_json():
    tr = OllamaTranslator()
    calls = {"n": 0}

    def fake_generate(body):
        calls["n"] += 1
        if "format" in body:          # the batch attempt -> return unparseable garbage
            return {"response": "not json"}
        return {"response": "폴백"}   # per-text fallback calls

    tr._generate = fake_generate
    out = tr.translate_batch(["長い文章その一", "長い文章その二"], "ja", "ko", {"model": "m"})
    assert out == ["폴백", "폴백"]     # fallback filled both, aligned
    assert calls["n"] == 3            # 1 failed batch + 2 per-text


TESTS = [
    test_builds_request_from_tuned_config,
    test_options_override,
    test_short_text_skips_model_call,
    test_missing_model_raises,
    test_batch_builds_format_request_and_aligns,
    test_batch_passes_through_short_texts,
    test_batch_falls_back_to_per_text_on_bad_json,
]

if __name__ == "__main__":
    import sys

    from scanlation_sdk.testing import run

    sys.exit(run(TESTS, "test_ollama"))
