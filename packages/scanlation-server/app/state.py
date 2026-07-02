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
    # Active LLM system-prompt preset name (see app.prompts) + user-saved presets.
    prompt_active: str = "default"
    prompts: dict[str, str] = field(default_factory=dict)


class AppState:
    """Process-wide selection + locks. One instance per process."""

    def __init__(self) -> None:
        self._path: Path = settings.data_dir / "state.json"
        self._lock = threading.Lock()
        self.selection = self._load()
        # Single GPU lock: detect + recognize share one device. Translation
        # (ollama) is a separate process and runs outside this lock so one image's
        # translate overlaps the next image's detect+recognize.
        self.gpu_lock = asyncio.Lock()
        # Bound concurrent translations (they run off the GPU lock) so many
        # in-flight images don't overrun the ollama backend's parallel slots.
        self.translate_sem = asyncio.Semaphore(settings.translate_concurrency)
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

    def set_options(self, engine_name: str, options: dict[str, Any]) -> None:
        """Persist per-engine option overrides (the admin 'engine options' form).

        A value of None or "" *removes* the override, so the field reverts to the
        engine's schema default (e.g. clearing the translator model unsets it).
        """
        cur = dict(self.selection.options.get(engine_name, {}))
        for key, val in (options or {}).items():
            if val is None or val == "":
                cur.pop(key, None)
            else:
                cur[key] = val
        if cur:
            self.selection.options[engine_name] = cur
        else:
            self.selection.options.pop(engine_name, None)
        self.save()

    def options_for(self, engine_name: str, request_options: dict | None) -> dict:
        """Merge persisted overrides with this request's options (request wins)."""
        merged = dict(self.selection.options.get(engine_name, {}))
        if request_options:
            merged.update(request_options.get(engine_name, {}) or {})
        return merged

    # --- LLM system prompt (shared by all LLM translators) ---
    def active_system_prompt(self) -> str:
        from .prompts import resolve_prompt

        return resolve_prompt(self.selection.prompt_active, self.selection.prompts)

    def translator_options(self, engine_name: str, request_options: dict | None) -> dict:
        """options_for + the active system prompt injected (unless overridden)."""
        opts = self.options_for(engine_name, request_options)
        opts.setdefault("system_prompt", self.active_system_prompt())
        return opts

    def save_prompt(self, name: str, text: str) -> None:
        """Upsert a user prompt preset and make it active."""
        self.selection.prompts[name] = text
        self.selection.prompt_active = name
        self.save()

    def select_prompt(self, name: str) -> None:
        self.selection.prompt_active = name
        self.save()

    def delete_prompt(self, name: str) -> None:
        self.selection.prompts.pop(name, None)
        if self.selection.prompt_active == name:
            self.selection.prompt_active = "default"
        self.save()


state = AppState()
