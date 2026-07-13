"""Processing-stats cache: raw page + per-crop rows -> summary with percentiles.

Runs against a throwaway base_dir (context.base_dir swapped + restored) so the real
scanlation.sqlite is never touched, and constructs a fresh ``Cache`` instance (never
the module singleton, whose connection bound the real base_dir at import).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from scanlation_sdk.context import context

from app.cache import Cache

from tests.helpers import run


def _fresh() -> Cache:
    context.base_dir = Path(tempfile.mkdtemp())   # a new dir -> a new sqlite file
    return Cache()


def _page(total_ms: float, regions: list[dict], *, skip: bool = False) -> dict:
    """record_stats kwargs for one page with the given per-crop rows."""
    return dict(
        page={"engines": "d+r+t", "src": "ja", "dst": "ko", "md5": "x",
              "regions": len(regions), "raw_regions": len(regions),
              "decode_ms": 5.0, "lockwait_ms": 0.0, "detect_ms": 10.0,
              "recognize_ms": 100.0, "semwait_ms": 0.0, "translate_ms": 0.0,
              "total_ms": total_ms},
        regions=regions, skip_translate=skip)


def _region(crop_w: int, rec_ms: float) -> dict:
    return {"crop_w": crop_w, "crop_h": 100, "score": 0.9,
            "source_len": 5, "dest_len": 6, "recognize_ms": rec_ms}


def test_record_and_summary():
    """Raw rows in, correct count + min/max/mean/median out for both tables."""
    saved = context.base_dir
    try:
        c = _fresh()
        c.record_stats(**_page(10.0, [_region(50, 100.0)]))
        c.record_stats(**_page(20.0, [_region(60, 200.0), _region(70, 300.0)]))
        c.record_stats(**_page(30.0, [_region(80, 400.0)]))
        s = c.stats_summary()
        assert s["pages"]["count"] == 3
        assert s["regions"]["count"] == 4                       # 1 + 2 + 1 crops
        tm = s["pages"]["metrics"]["total_ms"]
        assert (tm["min"], tm["max"], tm["mean"], tm["median"]) == (10.0, 30.0, 20.0, 20.0)
        rm = s["regions"]["metrics"]["recognize_ms"]
        assert rm["min"] == 100.0 and rm["max"] == 400.0
        assert "p90" in rm and "p99" in rm                      # percentiles present (>=2 rows)
    finally:
        context.base_dir = saved


def test_summary_empty():
    """Empty tables -> count 0, no crash, no metrics."""
    saved = context.base_dir
    try:
        c = _fresh()
        s = c.stats_summary()
        assert s["pages"]["count"] == 0 and s["pages"]["metrics"] == {}
        assert s["regions"]["count"] == 0 and s["regions"]["metrics"] == {}
    finally:
        context.base_dir = saved


def test_skip_translate_excluded():
    """A skip_translate (benchmark) page and its crops are filtered from the summary."""
    saved = context.base_dir
    try:
        c = _fresh()
        c.record_stats(**_page(10.0, [_region(50, 100.0)]))
        c.record_stats(**_page(99.0, [_region(50, 100.0)], skip=True))
        s = c.stats_summary()
        assert s["pages"]["count"] == 1                         # skip page excluded
        assert s["regions"]["count"] == 1                       # and its regions
    finally:
        context.base_dir = saved


def test_clear_stats():
    """clear_stats wipes both tables (returns pages + regions removed)."""
    saved = context.base_dir
    try:
        c = _fresh()
        c.record_stats(**_page(10.0, [_region(50, 100.0), _region(60, 200.0)]))
        assert c.clear_stats() == 3                             # 1 page + 2 regions
        assert c.stats_summary()["pages"]["count"] == 0
        assert c.stats_summary()["regions"]["count"] == 0
    finally:
        context.base_dir = saved


TESTS = [
    test_record_and_summary,
    test_summary_empty,
    test_skip_translate_excluded,
    test_clear_stats,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_cache"))
