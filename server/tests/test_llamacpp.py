"""LlamaCppTranslator unit tests — OpenAI-compatible request shape, HTTP mocked."""
from __future__ import annotations

import pytest

from plugins.translator_llamacpp.plugin import LlamaCppTranslator


@pytest.fixture
def translator(monkeypatch):
    tr = LlamaCppTranslator()
    captured: dict = {}

    def fake_chat(body):
        captured.clear()
        captured.update(body)
        return {"choices": [{"message": {"content": "  <think>음...</think>안녕하세요  "}}]}

    monkeypatch.setattr(tr, "_chat", fake_chat)
    tr._captured = captured
    return tr


def test_builds_openai_chat_request(translator):
    out = translator.translate("こんにちは", "ja", "ko", {})
    assert out == "안녕하세요"  # stripped + <think> removed

    b = translator._captured
    assert b["stream"] is False
    sys_msg, user_msg = b["messages"]
    assert sys_msg["role"] == "system" and sys_msg["content"].startswith("From now on")
    assert user_msg["role"] == "user"
    assert 'src="japanese"' in user_msg["content"]
    assert 'dst="korean"' in user_msg["content"]
    assert 'text="こんにちは"' in user_msg["content"]
    assert b["temperature"] == 0.0 and b["seed"] == 42 and b["top_p"] == 1.0 and b["max_tokens"] == 512


def test_keep_think_when_disabled(monkeypatch):
    tr = LlamaCppTranslator()
    monkeypatch.setattr(tr, "_chat", lambda body: {"choices": [{"message": {"content": "<think>x</think>네"}}]})
    out = tr.translate("テスト文章", "ja", "ko", {"strip_think": False})
    assert "<think>" in out


def test_short_text_skips(monkeypatch):
    tr = LlamaCppTranslator()
    called = False

    def fake(body):
        nonlocal called
        called = True
        return {"choices": [{"message": {"content": "x"}}]}

    monkeypatch.setattr(tr, "_chat", fake)
    assert tr.translate("あ", "ja", "ko", {}) == "あ"
    assert called is False
