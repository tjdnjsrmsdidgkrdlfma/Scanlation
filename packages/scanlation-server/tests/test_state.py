"""state.json persistence — the save/load round trip behind AppState.

Runs against a throwaway base_dir (context.base_dir is swapped in and restored)
so the developer's real data/state.json is never touched.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scanlation_sdk.context import context

# Import at module scope, NOT inside the tests: importing app.state creates the
# process-wide singleton, which must capture the real base_dir — not the
# temporary one these tests swap in.
from app.state import AppState, Selection

from tests.helpers import run


def test_state_json_roundtrip():
    saved_base = context.base_dir
    try:
        context.base_dir = Path(tempfile.mkdtemp())  # settings.data_dir delegates here
        st = AppState()
        st.set_engines("det-x", "rec-x", "tr-x")
        st.set_languages("en", "ja")
        st.set_engine_device("rec-x", "cuda")
        st.set_recognize_concurrency("rec-x", 4)
        st.set_options("x", {"a": 1})
        st.save_prompt("mine", "PROMPT")
        old_sem = st.translate_sem
        st.set_client_config(min_image_dim=123, verbose_log=True, translate_concurrency=8,
                             model_idle_unload_minutes=20,
                             torch_backend="gpu", torch_vendor="amd", torch_index="https://x/rocm6.2")
        assert st.translate_sem is not old_sem  # semaphore instance swapped at runtime
        # a fresh instance reads state.json back; dataclass equality covers every field
        assert AppState().selection == st.selection
        assert AppState().selection.recognize_concurrency == {"rec-x": 4}  # per-engine pool size persisted
        assert AppState().selection.verbose_log is True  # verbose toggle persisted
        assert AppState().selection.translate_concurrency == 8  # concurrency persisted
        assert AppState().selection.model_idle_unload_minutes == 20  # idle-unload window persisted
        assert AppState().selection.torch_backend == "gpu"      # torch backend persisted
        assert AppState().selection.torch_vendor == "amd"
        assert AppState().selection.torch_index == "https://x/rocm6.2"
    finally:
        context.base_dir = saved_base


def test_state_load_falls_back_on_bad_json():
    saved_base = context.base_dir
    try:
        context.base_dir = Path(tempfile.mkdtemp())
        data_dir = context.base_dir / "data"
        data_dir.mkdir(parents=True)
        path = data_dir / "state.json"
        path.write_text("{not json", encoding="utf-8")
        assert AppState().selection == Selection()  # decode error -> defaults
        path.write_text(json.dumps({"unknown_key": 1}), encoding="utf-8")
        assert AppState().selection == Selection()  # unknown field -> TypeError -> defaults
    finally:
        context.base_dir = saved_base


def test_engine_device_override():
    saved_base = context.base_dir
    try:
        context.base_dir = Path(tempfile.mkdtemp())
        st = AppState()
        assert st.resolve_device_for("comic-text-and-bubble-detector") is None  # no override -> engine default
        st.set_engine_device("comic-text-and-bubble-detector", "cuda")
        assert st.resolve_device_for("comic-text-and-bubble-detector") == "cuda"
        st.set_engine_device("comic-text-and-bubble-detector", "")              # blank removes it
        assert st.resolve_device_for("comic-text-and-bubble-detector") is None
        assert "comic-text-and-bubble-detector" not in st.selection.devices
    finally:
        context.base_dir = saved_base


def test_recognize_concurrency_override():
    """Per-engine recognize worker-pool size mirrors the device override: absent ->
    the global default; an explicit int is stored (incl. 1 to force 'no pool' even if
    the global default is higher); None resets to the default."""
    from app.config import settings

    saved_base = context.base_dir
    try:
        context.base_dir = Path(tempfile.mkdtemp())
        st = AppState()
        eng = "paddleocr-vl-for-manga"
        assert st.resolve_recognize_concurrency(eng) == max(1, settings.recognize_concurrency)  # no override
        st.set_recognize_concurrency(eng, 4)
        assert st.resolve_recognize_concurrency(eng) == 4
        assert st.selection.recognize_concurrency[eng] == 4
        st.set_recognize_concurrency(eng, 1)                    # explicit 1 forces no pool, is kept
        assert st.resolve_recognize_concurrency(eng) == 1
        assert st.selection.recognize_concurrency[eng] == 1
        st.set_recognize_concurrency(eng, None)                 # None resets to the global default
        assert eng not in st.selection.recognize_concurrency
        assert st.resolve_recognize_concurrency(eng) == max(1, settings.recognize_concurrency)
    finally:
        context.base_dir = saved_base


def test_config_env_seeds_settings_and_selection():
    """B-grade values now carry an env default: Settings() reads the env (with a
    floor-1 guard on translate_concurrency), and Selection seeds these fields from
    the settings singleton rather than a bare literal."""
    import os

    from app.config import Settings, settings

    os.environ["SCANLATION_TRANSLATE_CONCURRENCY"] = "5"
    try:
        assert Settings().translate_concurrency == 5           # env read per instance
        os.environ["SCANLATION_TRANSLATE_CONCURRENCY"] = "0"
        assert Settings().translate_concurrency == 1           # floor: a 0 Semaphore would deadlock
    finally:
        os.environ.pop("SCANLATION_TRANSLATE_CONCURRENCY", None)

    os.environ["SCANLATION_MODEL_IDLE_UNLOAD_MINUTES"] = "12"
    try:
        assert Settings().model_idle_unload_minutes == 12      # env read per instance
        os.environ["SCANLATION_MODEL_IDLE_UNLOAD_MINUTES"] = "-1"
        assert Settings().model_idle_unload_minutes == 0       # floor 0 (0 = never unload)
    finally:
        os.environ.pop("SCANLATION_MODEL_IDLE_UNLOAD_MINUTES", None)

    os.environ["SCANLATION_RECOGNIZE_CONCURRENCY"] = "4"
    try:
        assert Settings().recognize_concurrency == 4           # env read per instance
        os.environ["SCANLATION_RECOGNIZE_CONCURRENCY"] = "0"
        assert Settings().recognize_concurrency == 1           # floor 1 (1 = no pool)
    finally:
        os.environ.pop("SCANLATION_RECOGNIZE_CONCURRENCY", None)

    # Selection defaults are seeded from the settings singleton (wiring, not literals)
    sel = Selection()
    assert sel.translate_concurrency == settings.translate_concurrency
    assert sel.model_idle_unload_minutes == settings.model_idle_unload_minutes
    assert sel.torch_backend == settings.torch_backend
    assert sel.torch_vendor == settings.torch_vendor
    assert sel.torch_index == settings.torch_index
    assert sel.recognize_concurrency == {}                 # per-engine overrides start empty


TESTS = [
    test_state_json_roundtrip,
    test_state_load_falls_back_on_bad_json,
    test_engine_device_override,
    test_recognize_concurrency_override,
    test_config_env_seeds_settings_and_selection,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_state"))
