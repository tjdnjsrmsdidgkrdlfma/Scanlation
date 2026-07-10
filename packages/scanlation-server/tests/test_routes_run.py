"""routes/run.py wire-protocol tests via dummy engines (zero model risk).
Proves the md5/bounds/lazy contract for /run_pipeline/ and /run_lookup/.
"""
from __future__ import annotations

from app.pipeline import ResultItem
from tests.helpers import client, payload, run


def test_run_pipeline_work_returns_result_items():
    p = payload()
    r = client().post("/run_pipeline/", json={"md5": p["md5"], "contents": p["b64"]})
    assert r.status_code == 200
    result = r.json()["result"]
    assert len(result) == 2
    for item in result:
        # Assert against ResultItem itself, so renaming a key there can't leave the
        # extension reading a shape nothing declares.
        assert set(item) == set(ResultItem.__annotations__)
        assert len(item["bounds"]) == 4


def test_run_pipeline_md5_mismatch_is_400():
    p = payload()
    r = client().post("/run_pipeline/", json={"md5": "deadbeef", "contents": p["b64"]})
    assert r.status_code == 400


def test_no_engine_installed_is_400():
    """The core ships no engine; running a role with none selected -> 400."""
    from app.state import state

    c = client()
    saved = (state.selection.detector, state.selection.recognizer, state.selection.translator)
    try:
        state.selection.detector = ""                 # no detector installed/selected
        p = payload(color=(7, 7, 7))                  # unique md5 -> cache miss -> runs
        r = c.post("/run_pipeline/", json={"md5": p["md5"], "contents": p["b64"]})
        assert r.status_code == 400
    finally:
        state.selection.detector, state.selection.recognizer, state.selection.translator = saved


def test_lookup_miss_then_work_then_hit():
    c = client()
    p = payload(color=(123, 222, 31))  # unique md5

    # lookup miss -> 200 {result: null} (a cache probe, not a 404 control signal)
    miss = c.post("/run_lookup/", json={"md5": p["md5"]})
    assert miss.status_code == 200 and miss.json()["result"] is None

    # work populates the cache
    work = c.post("/run_pipeline/", json={"md5": p["md5"], "contents": p["b64"]})
    assert work.status_code == 200

    # lookup again -> served from cache, same result (no image re-upload)
    hit = c.post("/run_lookup/", json={"md5": p["md5"]})
    assert hit.status_code == 200
    assert hit.json()["result"] == work.json()["result"]


def test_run_pipeline_requires_contents():
    """run_pipeline is work-only now; probing the cache without contents is /run_lookup/."""
    c = client()
    p = payload(color=(5, 6, 7))
    assert c.post("/run_pipeline/", json={"md5": p["md5"]}).status_code == 400


def test_run_pipeline_returns_per_stage_timing():
    """A fresh run carries the per-stage `timing` breakdown (headless tools read it);
    a cache hit does not, since run_page — where timing is measured — is skipped."""
    c = client()
    p = payload(color=(9, 42, 200))  # unique md5 -> cache miss -> fresh, timed run
    fresh = c.post("/run_pipeline/", json={"md5": p["md5"], "contents": p["b64"]})
    assert fresh.status_code == 200
    timing = fresh.json().get("timing")
    assert timing is not None, "a fresh run must carry timing"
    assert set(timing) == {
        "decode_ms", "lockwait_ms", "detect_recognize_ms", "detect_ms", "recognize_ms",
        "semwait_ms", "translate_ms", "total_ms", "regions",
    }
    assert all(isinstance(v, (int, float)) for v in timing.values())

    # same md5, no force -> served from cache -> run_page skipped -> no timing key
    hit = c.post("/run_pipeline/", json={"md5": p["md5"], "contents": p["b64"]})
    assert hit.status_code == 200
    assert "timing" not in hit.json()


TESTS = [
    test_run_pipeline_work_returns_result_items,
    test_run_pipeline_md5_mismatch_is_400,
    test_no_engine_installed_is_400,
    test_lookup_miss_then_work_then_hit,
    test_run_pipeline_requires_contents,
    test_run_pipeline_returns_per_stage_timing,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_routes_run"))
