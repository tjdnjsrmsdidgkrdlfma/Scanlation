"""Mutable selection state + runtime concurrency primitives.

Selection (which engine per role, src/dst langs, per-engine option overrides) is
persisted to a small json so restarts keep the user's choice. The GPU lock and
in-flight dedupe map are runtime-only.
"""
from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import settings


@dataclass
class Selection:
    detector: str = settings.default_detector
    recognizer: str = settings.default_recognizer
    translator: str = settings.default_translator
    lang_src: str = settings.default_lang_src
    lang_dst: str = settings.default_lang_dst
    # {engine_name: {opt: val}} overrides applied on top of schema defaults.
    options: dict[str, dict[str, Any]] = field(default_factory=dict)


class AppState:
    """Process-wide selection + locks. One instance per process."""

    def __init__(self) -> None:
        self._path: Path = settings.data_dir / "state.json"
        self._lock = threading.Lock()
        self.selection = self._load()
        # Single GPU lock: detect + recognize share one device. Translation
        # (ollama) is a separate process and is not guarded here.
        self.gpu_lock = asyncio.Lock()
        # md5/opts identity -> in-flight Future, so duplicate concurrent
        # requests for the same image attach to one computation.
        self.inflight: dict[tuple, asyncio.Future] = {}

    def _load(self) -> Selection:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return Selection(**data)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return Selection()

    def save(self) -> None:
        with self._lock:
            settings.ensure_dirs()
            self._path.write_text(json.dumps(asdict(self.selection), ensure_ascii=False, indent=2), encoding="utf-8")

    # --- mutations (validated against the registry by the caller) ---
    def set_models(self, detector: str | None, recognizer: str | None, translator: str | None) -> None:
        if detector:
            self.selection.detector = detector
        if recognizer:
            self.selection.recognizer = recognizer
        if translator:
            self.selection.translator = translator
        self.save()

    def set_langs(self, lang_src: str, lang_dst: str) -> None:
        self.selection.lang_src = lang_src
        self.selection.lang_dst = lang_dst
        self.save()

    def options_for(self, engine_name: str, request_options: dict | None) -> dict:
        """Merge persisted overrides with this request's options (request wins)."""
        merged = dict(self.selection.options.get(engine_name, {}))
        if request_options:
            merged.update(request_options.get(engine_name, {}) or {})
        return merged


state = AppState()
