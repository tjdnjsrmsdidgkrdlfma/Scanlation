"""Static configuration + environment settings.

No pydantic-settings dependency: a small dataclass read from env keeps the core
import-light. The plugin-facing slice (models_dir, language tables) lives
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

    # First-run default engine selection (role -> engine name) + languages; the
    # admin page overrides these into state.json afterwards. detector defaults to
    # comic-text-and-bubble-detector (the chosen default detector); recognizer/
    # translator stay empty until installed and picked (running a role with none
    # installed/selected is a 400).
    default_detector: str = field(default_factory=lambda: _env("SCANLATION_DETECTOR", "comic-text-and-bubble-detector"))
    default_recognizer: str = field(default_factory=lambda: _env("SCANLATION_RECOGNIZER", ""))
    default_translator: str = field(default_factory=lambda: _env("SCANLATION_TRANSLATOR", ""))
    default_lang_src: str = field(default_factory=lambda: _env("SCANLATION_LANG_SRC", "ja"))
    default_lang_dst: str = field(default_factory=lambda: _env("SCANLATION_LANG_DST", "ko"))

    # Shared secret gating the API/admin (sent as the X-Auth-Token header). Empty
    # = no auth (local/dev; the current default). Set it to lock a public deploy.
    auth_token: str = field(default_factory=lambda: _env("SCANLATION_AUTH_TOKEN", ""))

    # Log level for the app's own loggers (scanlation.*). Third-party libs stay at
    # WARNING (root) so transformers/httpx don't drown the log. See app.logconfig.
    log_level: str = field(default_factory=lambda: _env("SCANLATION_LOG_LEVEL", "INFO"))

    # First-run default for the extension's image filter: images whose SHORTER
    # side is under this (px) are skipped as icons/banners. Persisted per-install
    # in state.json and editable in /admin (동작 tab); delivered to the extension
    # via the handshake. 0 = translate everything.
    min_image_dim: int = field(
        default_factory=lambda: int(_env("SCANLATION_MIN_IMAGE_DIM", "80"))
    )

    # --- filesystem: delegated to the shared SDK context (single env source of
    #     truth, also read by every engine plugin) ---
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
