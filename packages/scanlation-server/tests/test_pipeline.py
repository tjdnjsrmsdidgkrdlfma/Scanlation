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
        opt_box={}, opt_ocr={}, opt_tsl={},
    )
    assert len(result) == 2
    assert result[0]["ocr"] == "REGION-0"
    assert result[0]["tsl"] == "[ja->ko] REGION-0"
    assert result[1]["ocr"] == "REGION-1"
    for item in result:
        assert len(item["box"]) == 4
        x0, y0, x1, y1 = item["box"]
        assert 0 <= x0 < x1 <= 400 and 0 <= y0 < y1 <= 300


TESTS = [
    test_reading_order_is_right_to_left_top_to_bottom,
    test_dummy_pipeline_golden,
]

if __name__ == "__main__":
    import sys

    from tests.helpers import run

    sys.exit(run(TESTS, "test_pipeline"))
