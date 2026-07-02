"""Static configuration + environment settings.

No pydantic-settings dependency: a small dataclass read from env keeps the core
import-light. The plugin-facing slice (models_dir, device, language tables) lives
in ``scanlation_sdk.context`` — the single source shared with engine plugins;
this module delegates to it so there's no drift. The handshake never loads a
model, so config must not depend on any engine.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from scanlation_sdk.context import context


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    host: str = field(default_factory=lambda: _env("SCANLATION_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("SCANLATION_PORT", "4000")))

    # First-run default engine selection (role -> plugin name) + languages; the
    # admin page overrides these into state.json afterwards. Empty by default:
    # the core ships no engine, so nothing is selected until one is installed and
    # picked (running without one is a 400).
    default_detector: str = field(default_factory=lambda: _env("SCANLATION_DETECTOR", ""))
    default_recognizer: str = field(default_factory=lambda: _env("SCANLATION_RECOGNIZER", ""))
    default_translator: str = field(default_factory=lambda: _env("SCANLATION_TRANSLATOR", ""))
    default_lang_src: str = field(default_factory=lambda: _env("SCANLATION_LANG_SRC", "ja"))
    default_lang_dst: str = field(default_factory=lambda: _env("SCANLATION_LANG_DST", "ko"))

    # --- filesystem + device: delegated to the shared SDK context (single env
    #     source of truth, also read by every engine plugin) ---
    @property
    def device(self) -> str:
        return context.device

    @property
    def base_dir(self) -> Path:
        return context.base_dir

    @property
    def models_dir(self) -> Path:
        return context.models_dir

    @property
    def data_dir(self) -> Path:
        return context.base_dir / "data"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
