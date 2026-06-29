"""OllamaTranslator unit tests — request shape + parsing, HTTP mocked.

ollama itself runs on a separate (Linux/ROCm) host, so these never hit the
network: _generate is monkeypatched to capture the request body.
"""
from __future__ import annotations

import pytest

from plugins.translator_ollama.plugin import OllamaTranslator


@pytest.fixture
def translator(monkeypatch):
    tr = OllamaTranslator()
    captured: dict = {}

    def fake_generate(body):
        captured.clear()
        captured.update(body)
        return {"response": "  안녕하세요  ", "done": True}

    monkeypatch.setattr(tr, "_generate", fake_generate)
    tr._captured = captured
    return tr


def test_builds_request_from_tuned_config(translator):
    out = translator.translate("こんにちは", "ja", "ko", {})
    assert out == "안녕하세요"  # response stripped

    body = translator._captured
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


def test_options_override(translator):
    translator.translate("テスト文章です", "ja", "ko", {"num_ctx": 1024, "think": True, "temperature": 0.7})
    body = translator._captured
    assert body["think"] is True
    assert body["options"]["num_ctx"] == 1024
    assert body["options"]["temperature"] == 0.7


def test_short_text_skips_model_call(monkeypatch):
    tr = OllamaTranslator()
    called = False

    def fake(body):
        nonlocal called
        called = True
        return {"response": "x"}

    monkeypatch.setattr(tr, "_generate", fake)
    assert tr.translate("あ", "ja", "ko", {}) == "あ"  # <=2 chars returned as-is
    assert tr.translate("  ", "ja", "ko", {}) == ""
    assert called is False
