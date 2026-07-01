"""LlamaCppTranslator unit tests — OpenAI-compatible request shape, HTTP mocked."""
from __future__ import annotations

from plugins.translator_llamacpp.plugin import LlamaCppTranslator


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
    assert sys_msg["role"] == "system" and sys_msg["content"].startswith("From now on")
    assert user_msg["role"] == "user"
    assert 'src="japanese"' in user_msg["content"]
    assert 'dst="korean"' in user_msg["content"]
    assert 'text="こんにちは"' in user_msg["content"]
    assert b["temperature"] == 0.0 and b["seed"] == 42 and b["top_p"] == 1.0 and b["max_tokens"] == 512


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


TESTS = [
    test_builds_openai_chat_request,
    test_keep_think_when_disabled,
    test_missing_model_raises,
    test_short_text_skips,
]

if __name__ == "__main__":
    import sys

    from tests.helpers import run

    sys.exit(run(TESTS, "test_llamacpp"))
