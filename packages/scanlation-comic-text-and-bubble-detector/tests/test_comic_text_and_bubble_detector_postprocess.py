"""RT-DETR post-processing — testable with no torch/transformers/weights."""
from __future__ import annotations

from scanlation_comic_text_and_bubble_detector.postprocess import Det, dedup, filter_labels, ios, iou, to_regions


def _det(x0, y0, x1, y1, label="text_free", score=0.9):
    return Det((x0, y0, x1, y1), label, score)


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-6


def test_ios_catches_nesting():
    big, small = (0, 0, 100, 100), (40, 40, 60, 60)  # small fully inside big
    assert iou(big, small) < 0.05   # IoU misses it (union is huge)
    assert ios(big, small) == 1.0   # IoS catches it (inter / smaller area)


def test_filter_labels_drops_bubble():
    dets = [_det(0, 0, 10, 10, "bubble"), _det(0, 0, 10, 10, "text_bubble"), _det(0, 0, 10, 10, "text_free")]
    kept = filter_labels(dets, {"text_bubble", "text_free"})
    assert {d.label for d in kept} == {"text_bubble", "text_free"}
    assert filter_labels(dets, None) == dets  # None = keep all


def test_dedup_removes_iou_duplicate():
    dets = [_det(0, 0, 100, 20, score=0.9), _det(1, 1, 99, 21, score=0.5)]  # near-identical
    kept = dedup(dets, nms_iou=0.6, contain_thresh=0.85)
    assert len(kept) == 1 and kept[0].score == 0.9  # higher score survives


def test_dedup_removes_nested_small_box():
    big = _det(0, 0, 100, 100, score=0.9)
    small = _det(40, 40, 60, 60, score=0.5)  # nested; low IoU with big, high IoS
    kept = dedup([big, small], nms_iou=0.6, contain_thresh=0.85)
    assert kept == [big]  # nested one dropped via IoS


def test_dedup_keeps_disjoint():
    a, b = _det(0, 0, 10, 10, score=0.9), _det(50, 50, 70, 70, score=0.8)
    assert len(dedup([a, b], nms_iou=0.6, contain_thresh=0.85)) == 2


def test_dedup_off_when_thresholds_none():
    dets = [_det(0, 0, 100, 20, score=0.9), _det(1, 1, 99, 21, score=0.5)]
    assert len(dedup(dets, None, None)) == 2


def test_to_regions_axis_aligned_and_vertical_flag():
    regions = to_regions([_det(10, 20, 40, 90, score=0.7)])  # taller than wide -> vertical
    assert len(regions) == 1
    r = regions[0]
    assert r.bbox == (10, 20, 40, 90)
    assert r.polygon.shape == (4, 2)
    assert r.angle == 0.0
    assert r.vertical is True
    assert abs(r.score - 0.7) < 1e-6
    assert r.label == "text_free"  # detector class carried onto the region
    assert to_regions([_det(0, 0, 10, 5, label="text_bubble")])[0].label == "text_bubble"


TESTS = [
    test_iou_basic,
    test_ios_catches_nesting,
    test_filter_labels_drops_bubble,
    test_dedup_removes_iou_duplicate,
    test_dedup_removes_nested_small_box,
    test_dedup_keeps_disjoint,
    test_dedup_off_when_thresholds_none,
    test_to_regions_axis_aligned_and_vertical_flag,
]

if __name__ == "__main__":
    import sys

    from scanlation_sdk.testing import run

    sys.exit(run(TESTS, "test_comic_text_and_bubble_detector_postprocess"))
