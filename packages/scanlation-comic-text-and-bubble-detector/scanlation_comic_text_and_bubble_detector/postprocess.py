"""RT-DETR post-processing — deliberately model-free (no torch/transformers) so
the geometry is unit-testable with plain data.

RT-DETR gives one detection per query as an axis-aligned xyxy box with a class
(bubble / text_bubble / text_free). For recognition we: keep only the text classes (drop
the whole-bubble container), dedup the overlapping/nested boxes RT-DETR leaves
behind (it's NMS-free, so duplicates survive), and map the survivors to Regions.

Kept separate from plugin.py so ``python -m tests`` exercises it without weights.
"""
from __future__ import annotations

from dataclasses import dataclass

from scanlation_sdk.contracts import Region


@dataclass
class Det:
    """One raw detection in original-image pixels."""
    xyxy: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    label: str
    score: float


def iou(a: tuple, b: tuple) -> float:
    """Intersection over union of two xyxy boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ios(a: tuple, b: tuple) -> float:
    """Intersection over the *smaller* box's area — catches a small box nested
    inside a big one (their IoU is low, but IoS is high)."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def filter_labels(dets: list[Det], keep_labels: set[str] | None) -> list[Det]:
    """Keep only detections whose label is in keep_labels (None = keep all)."""
    if keep_labels is None:
        return list(dets)
    return [d for d in dets if d.label in keep_labels]


def dedup(dets: list[Det], nms_iou: float | None, contain_thresh: float | None) -> list[Det]:
    """Greedy suppression, highest score first: drop a detection that overlaps an
    already-kept (higher-score) one past ``nms_iou`` (IoU, same-size duplicates)
    OR sits ``contain_thresh`` inside/around one (IoS, nested small-in-big). Either
    threshold None disables that half; 1.0 makes it effectively off."""
    if not dets or (nms_iou is None and contain_thresh is None):
        return list(dets)
    kept: list[Det] = []
    for d in sorted(dets, key=lambda x: -x.score):
        if any(
            (nms_iou is not None and iou(d.xyxy, k.xyxy) >= nms_iou)
            or (contain_thresh is not None and ios(d.xyxy, k.xyxy) >= contain_thresh)
            for k in kept
        ):
            continue
        kept.append(d)
    return kept


def to_regions(dets: list[Det]) -> list[Region]:
    """Map xyxy detections to Regions. RT-DETR boxes are axis-aligned (angle 0);
    a taller-than-wide box is flagged vertical (Japanese vertical writing)."""
    out: list[Region] = []
    for d in dets:
        x0, y0, x1, y1 = d.xyxy
        out.append(Region.from_bbox(x0, y0, x1, y1, score=d.score, vertical=(y1 - y0) > (x1 - x0), label=d.label))
    return out
