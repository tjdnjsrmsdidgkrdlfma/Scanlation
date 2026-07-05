"""LlamaCppTranslator unit tests — OpenAI-compatible request shape, HTTP mocked."""
from __future__ import annotations

import json

from scanlation_llama_cpp.plugin import LlamaCppTranslator


def _translator() -> LlamaCppTranslator:
    """A LlamaCppTranslator whose _chat is faked; tr._captured holds the body."""
    tr = LlamaCppTranslator()
    captured: dict = {}

    def fake_chat(body):
        captured.clear()
        captured.update(body)
        return {"choices": [{"message": {"content": "  <think>음...</think>안녕하세요  "}}]}

    tr._chat = fake_chat
    tr._captured = captured
    return tr


def test_builds_openai_chat_request():
    translator = _translator()
    out = translator.translate("こんにちは", "ja", "ko", {"model": "local-test"})
    assert out == "안녕하세요"  # stripped + <think> removed

    b = translator._captured
    assert b["model"] == "local-test"  # model from options (admin), no env fallback
    assert b["stream"] is False
    sys_msg, user_msg = b["messages"]
    assert sys_msg["role"] == "system" and sys_msg["content"].startswith("You are a translator")
    assert user_msg["role"] == "user"
    assert 'src="japanese"' in user_msg["content"]
    assert 'dst="korean"' in user_msg["content"]
    assert 'text="こんにちは"' in user_msg["content"]
    assert b["temperature"] == 0.0 and b["seed"] == 42 and b["top_p"] == 1.0
    assert "max_tokens" not in b  # no explicit output cap; model stops at EOS


def test_keep_think_when_disabled():
    tr = LlamaCppTranslator()
    tr._chat = lambda body: {"choices": [{"message": {"content": "<think>x</think>네"}}]}
    out = tr.translate("テスト文章", "ja", "ko", {"model": "local-test", "strip_think": False})
    assert "<think>" in out


def test_missing_model_raises():
    tr = LlamaCppTranslator()
    tr._chat = lambda body: {"choices": [{"message": {"content": "x"}}]}  # never reached
    raised = False
    try:
        tr.translate("これは十分に長い文章です", "ja", "ko", {})  # no model in options
    except ValueError:
        raised = True
    assert raised, "translate must raise when no model is selected"


def test_short_text_skips():
    tr = LlamaCppTranslator()
    called = False

    def fake(body):
        nonlocal called
        called = True
        return {"choices": [{"message": {"content": "x"}}]}

    tr._chat = fake
    assert tr.translate("あ", "ja", "ko", {}) == "あ"
    assert called is False


def test_batch_builds_response_format_and_aligns():
    tr = LlamaCppTranslator()
    captured: dict = {}

    def fake_chat(body):
        captured.clear()
        captured.update(body)
        return {"choices": [{"message": {"content": json.dumps({"t0": "가", "t1": "나"})}}]}

    tr._chat = fake_chat
    out = tr.translate_batch(["日本語一", "日本語二"], "ja", "ko", {"model": "m"})
    assert out == ["가", "나"]
    rf = captured["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"]["required"] == ["t0", "t1"]  # exactly 2 keys forced
    assert "max_tokens" not in captured                             # no cap; JSON grammar bounds output


def test_batch_falls_back_on_wrong_length():
    tr = LlamaCppTranslator()

    def fake_chat(body):
        if "response_format" in body:  # batch attempt returns too few translations
            return {"choices": [{"message": {"content": json.dumps({"t0": "only one"})}}]}
        return {"choices": [{"message": {"content": "fb"}}]}  # per-text fallback

    tr._chat = fake_chat
    out = tr.translate_batch(["長い文章その一", "長い文章その二"], "ja", "ko", {"model": "m"})
    assert out == ["fb", "fb"]  # missing t1 -> fallback fills both, aligned


TESTS = [
    test_builds_openai_chat_request,
    test_keep_think_when_disabled,
    test_missing_model_raises,
    test_short_text_skips,
    test_batch_builds_response_format_and_aligns,
    test_batch_falls_back_on_wrong_length,
]

if __name__ == "__main__":
    import sys

    from scanlation_sdk.testing import run

    sys.exit(run(TESTS, "test_llama_cpp"))
