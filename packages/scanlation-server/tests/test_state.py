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
        st.set_engines("det-x", "rec-x", "tsl-x")
        st.set_languages("en", "ja")
        st.set_engine_device("rec-x", "cuda")
        st.set_options("x", {"a": 1})
        st.save_prompt("mine", "PROMPT")
        st.set_client_config(min_image_dim=123)
        # a fresh instance reads state.json back; dataclass equality covers every field
        assert AppState().selection == st.selection
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


TESTS = [
    test_state_json_roundtrip,
    test_state_load_falls_back_on_bad_json,
    test_engine_device_override,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_state"))
