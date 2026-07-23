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


def _env_int(name: str, default: int, floor: int | None = None) -> int:
    """Env-backed int with an optional lower clamp. ``floor`` guards the values a
    0/negative would break (a Semaphore/pool size); leave it None where any int is
    valid. The /admin write path clamps the same values at runtime; this is just the
    first-run seed."""
    value = int(_env(name, str(default)))
    return value if floor is None else max(floor, value)


@dataclass
class Settings:
    host: str = field(default_factory=lambda: _env("SCANLATION_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("SCANLATION_PORT", "4010")))

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
        default_factory=lambda: _env_int("SCANLATION_MIN_IMAGE_DIM", 80)
    )

    # First-run default for the concurrent-translation limit (bounds parallel ollama
    # requests). Persisted in state.json, editable in /admin (동작 tab). Floor 1: a
    # 0/negative Semaphore would deadlock, matching set_client_config's clamp.
    translate_concurrency: int = field(
        default_factory=lambda: _env_int("SCANLATION_TRANSLATE_CONCURRENCY", 1, floor=1)
    )

    # First-run default for the recognize worker-pool size (per-engine, overridable
    # in /admin plugin options). Floor 1: 1 = no pool (the in-process per-crop loop,
    # byte-identical to the pre-pool path); >1 fans a page's crops across N worker
    # PROCESSES (each B=1) to fill the GPU idle a single request leaves. It's a
    # per-engine LOAD-TIME setting (pool built with N workers), so it lives in
    # Selection.recognize_concurrency (a dict, like devices), NOT as an OPTION_SCHEMA
    # option; this is only the global fallback when an engine has no override.
    recognize_concurrency: int = field(
        default_factory=lambda: _env_int("SCANLATION_RECOGNIZE_CONCURRENCY", 1, floor=1)
    )

    # First-run default for the gate size (per-recognizer, overridable in /admin plugin
    # options). Floor 1: 1 = serial detect+recognize (today's behavior, byte-identical);
    # >1 lets that many images run the GPU half at once so their crops fill the SHARED
    # recognize pool together (cross-image overlap), lifting the per-image crop ceiling.
    # Like recognize_concurrency it's a per-recognizer LOAD-TIME setting (sizes the
    # InferenceGate) stored in Selection.gpu_concurrency; this is only the global fallback.
    gpu_concurrency: int = field(
        default_factory=lambda: _env_int("SCANLATION_GPU_CONCURRENCY", 1, floor=1)
    )

    # First-run default for idle model unload (MINUTES): a local torch engine
    # (detector/recognizer) not used for this long is dropped from VRAM by a
    # background sweep, so it stops holding the GPU between reading sessions — the
    # in-process analog of ollama's OLLAMA_KEEP_ALIVE (translators are separate
    # processes, unaffected). Persisted in state.json, editable in /admin (동작 tab).
    # Floor 0; 0 = never auto-unload (keep resident).
    model_idle_unload_minutes: int = field(
        default_factory=lambda: _env_int("SCANLATION_MODEL_IDLE_UNLOAD_MINUTES", 5, floor=0)
    )

    # First-run defaults for the GPU/torch build a plugin install pulls, so a headless
    # deploy can pick the wheel via env instead of visiting /admin first. Persisted in
    # state.json, editable in /admin (동작 tab); the /admin write path validates the
    # values (torch_backend -> cpu/gpu, torch_vendor -> ""/amd/nvidia).
    torch_backend: str = field(default_factory=lambda: _env("SCANLATION_TORCH_BACKEND", "cpu"))
    torch_vendor: str = field(default_factory=lambda: _env("SCANLATION_TORCH_VENDOR", ""))
    torch_index: str = field(default_factory=lambda: _env("SCANLATION_TORCH_INDEX", ""))

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
