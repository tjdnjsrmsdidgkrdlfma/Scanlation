"""End-to-end pipeline with dummy engines -> deterministic golden output."""
from __future__ import annotations

from PIL import Image

from app.pipeline import assign_reading_order, run_pipeline
from scanlation_sdk.contracts import Region
from tests.fake_engines import DummyDetector, DummyRecognizer, DummyTranslator


def test_reading_order_is_right_to_left_top_to_bottom():
    # two boxes on the same row: right one must come first (manga R->L)
    left = Region.from_bbox(10, 10, 40, 30)
    right = Region.from_bbox(60, 12, 90, 32)
    ordered = assign_reading_order([left, right])
    assert ordered[0] is right and ordered[0].order == 0
    assert ordered[1] is left and ordered[1].order == 1


def test_dummy_pipeline_golden():
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    result = run_pipeline(
        img,
        detector=DummyDetector(),
        recognizer=DummyRecognizer(),
        translator=DummyTranslator(),
        src="ja", dst="ko",
        opt_detect={}, opt_recognize={}, opt_translate={},
    )
    assert len(result) == 2
    assert result[0]["source"] == "REGION-0"
    assert result[0]["destination"] == "[ja->ko] REGION-0"
    assert result[1]["source"] == "REGION-1"
    for item in result:
        assert len(item["bounds"]) == 4
        x0, y0, x1, y1 = item["bounds"]
        assert 0 <= x0 < x1 <= 400 and 0 <= y0 < y1 <= 300


class _BatchRecorder:
    """Translator exposing translate_batch — records which path the pipeline took
    and returns results aligned to input order (so we can assert ordering)."""
    name = "batchfake"

    def __init__(self):
        self.batch_calls = 0
        self.single_calls = 0

    def translate(self, text, src, dst, options):
        self.single_calls += 1
        return f"[one->{dst}] {text}"

    def translate_batch(self, texts, src, dst, options):
        self.batch_calls += 1
        return [f"[batch->{dst}] {t}" for t in texts]


def test_batch_path_used_when_available_and_order_preserved():
    # A translator with translate_batch must be driven via ONE batch call (not the
    # per-text loop), and the result order must still match reading order.
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    tr = _BatchRecorder()
    result = run_pipeline(
        img,
        detector=DummyDetector(), recognizer=DummyRecognizer(), translator=tr,
        src="ja", dst="ko", opt_detect={}, opt_recognize={}, opt_translate={},
    )
    assert len(result) == 2
    assert tr.batch_calls == 1 and tr.single_calls == 0  # one batch, no per-text
    assert result[0]["source"] == "REGION-0" and result[0]["destination"] == "[batch->ko] REGION-0"
    assert result[1]["source"] == "REGION-1" and result[1]["destination"] == "[batch->ko] REGION-1"


def test_no_batch_method_falls_back_to_per_text():
    # DummyTranslator has no translate_batch -> pipeline uses the per-text loop.
    assert not hasattr(DummyTranslator(), "translate_batch")
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    result = run_pipeline(
        img,
        detector=DummyDetector(), recognizer=DummyRecognizer(), translator=DummyTranslator(),
        src="ja", dst="ko", opt_detect={}, opt_recognize={}, opt_translate={},
    )
    assert result[0]["destination"] == "[ja->ko] REGION-0"  # dummy per-text echo


TESTS = [
    test_reading_order_is_right_to_left_top_to_bottom,
    test_dummy_pipeline_golden,
    test_batch_path_used_when_available_and_order_preserved,
    test_no_batch_method_falls_back_to_per_text,
]

if __name__ == "__main__":
    import sys

    from tests.helpers import run

    sys.exit(run(TESTS, "test_pipeline"))
