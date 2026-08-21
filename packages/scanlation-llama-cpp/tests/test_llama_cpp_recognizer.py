"""LlamaCppRecognizer unit tests — vision request shape + cap, HTTP mocked."""
from __future__ import annotations

import base64
import io

from PIL import Image

from scanlation_llama_cpp.recognizer import LlamaCppRecognizer
from scanlation_sdk.contracts import Region


def _rec(content: str = "  こんにちは  ") -> LlamaCppRecognizer:
    """A recognizer whose _post is faked; rec._captured holds the request body."""
    rec = LlamaCppRecognizer()
    captured: dict = {}

    def fake_post(path, body):
        captured.clear()
        captured.update({"_path": path, **body})
        return {"choices": [{"message": {"content": content}}]}

    rec._post = fake_post
    rec._captured = captured
    return rec


def _uploaded(body: dict) -> Image.Image:
    """Decode the image the body carries, so assertions can check what was sent."""
    url = body["messages"][0]["content"][0]["image_url"]["url"]
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def _crop(w: int, h: int) -> Image.Image:
    return Image.new("RGB", (w, h), "white")


def test_builds_vision_request():
    rec = _rec()
    crop = _crop(100, 50)
    out = rec.recognize(crop, Region.from_bbox(0, 0, 100, 50), {})
    assert out == "こんにちは"  # content trimmed

    b = rec._captured
    assert b["_path"] == "/v1/chat/completions"
    assert b["stream"] is False
    assert b["temperature"] == 0.0 and b["max_tokens"] == 1024
    assert "model" not in b  # blank model omitted, not sent as ""
    image_part, text_part = b["messages"][0]["content"]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert text_part["text"] == "OCR:"


def test_admin_options_are_crop_reading_only():
    """A recognizer's admin surface is how the CROP is read. Which model serves it, the
    prompt that model wants, the token ceiling and the decode temperature describe the
    llama-server, so they must not show up as /admin fields."""
    assert set(LlamaCppRecognizer.OPTION_SCHEMA) == {"max_pixels", "downscale_mode"}


def test_serving_details_come_from_env():
    """...and are settable, per deployment, through their env vars."""
    import os

    env = {
        "LLAMACPP_RECOGNIZE_MODEL": "vl-test",
        "SCANLATION_RECOGNIZE_PROMPT": "READ:",
        "SCANLATION_RECOGNIZE_MAX_TOKENS": "32",
        "SCANLATION_RECOGNIZE_TEMPERATURE": "0.5",
    }
    os.environ.update(env)
    try:
        rec = _rec()
        rec.recognize(_crop(10, 10), Region.from_bbox(0, 0, 10, 10), {})
        b = rec._captured
        assert b["model"] == "vl-test"
        assert b["messages"][0]["content"][1]["text"] == "READ:"
        assert b["max_tokens"] == 32 and b["temperature"] == 0.5
    finally:
        for k in env:
            os.environ.pop(k, None)


def test_cap_downscales_before_upload():
    """The pixel cap is applied to the crop that is UPLOADED (it shrinks the wire
    payload too), and a crop already under the cap is sent untouched."""
    rec = _rec()
    cap = LlamaCppRecognizer.OPTION_SCHEMA["max_pixels"]["default"]
    big = _crop(1000, 1000)  # 1,000,000 px -> well over any sane cap
    rec.recognize(big, Region.from_bbox(0, 0, 1000, 1000), {})
    sent = _uploaded(rec._captured)
    # `box` lands ON the cap rather than overshooting to a power-of-two fraction of
    # it, so the upload should fill most of the budget — not a quarter of it.
    assert 0.9 * cap <= sent.width * sent.height <= cap

    rec.recognize(_crop(100, 100), Region.from_bbox(0, 0, 100, 100), {})
    small = _uploaded(rec._captured)
    assert (small.width, small.height) == (100, 100)


def test_cap_off_uploads_full_size():
    rec = _rec()
    rec.recognize(_crop(600, 400), Region.from_bbox(0, 0, 600, 400), {"max_pixels": 0})
    sent = _uploaded(rec._captured)
    assert (sent.width, sent.height) == (600, 400)


def test_empty_content_is_empty_string():
    """A server that answers with null/blank content yields "" rather than raising —
    a crop with no readable text is normal, not an error."""
    rec = _rec(content="")
    assert rec.recognize(_crop(10, 10), Region.from_bbox(0, 0, 10, 10), {}) == ""


def test_endpoint_from_env(monkeypatch=None):
    """Endpoint comes from LLAMACPP_RECOGNIZE_ENDPOINT, defaulting to its own port
    (separate llama-server from the translator's)."""
    import os

    assert LlamaCppRecognizer().endpoint == "http://127.0.0.1:8090"
    os.environ["LLAMACPP_RECOGNIZE_ENDPOINT"] = "http://example:9/"
    try:
        assert LlamaCppRecognizer().endpoint == "http://example:9"  # trailing slash stripped
    finally:
        del os.environ["LLAMACPP_RECOGNIZE_ENDPOINT"]


TESTS = [
    test_builds_vision_request,
    test_admin_options_are_crop_reading_only,
    test_serving_details_come_from_env,
    test_cap_downscales_before_upload,
    test_cap_off_uploads_full_size,
    test_empty_content_is_empty_string,
    test_endpoint_from_env,
]

if __name__ == "__main__":
    import sys

    from scanlation_sdk.testing import run

    sys.exit(run(TESTS, "test_llama_cpp_recognizer"))
