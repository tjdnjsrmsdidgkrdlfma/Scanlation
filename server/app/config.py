"""Static configuration + environment settings.

No pydantic-settings dependency: a small dataclass read from env keeps the
core import-light. Language table is a static iso1 map (handshake never loads
a model, so this must not depend on any engine).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# iso1 -> human readable. Source language is Japanese, target Korean, but we
# expose a small useful set so the popup language dropdowns render.
LANGUAGES: dict[str, str] = {
    "ja": "Japanese",
    "ko": "Korean",
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}

# iso1 -> plain language name passed to translators (ollama wants "japanese").
LANG_PLAIN: dict[str, str] = {k: v.lower() for k, v in LANGUAGES.items()}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # Compute device hint for engines: cpu | rocm | dml. Engines always keep a
    # CPU fallback; this only selects the preferred onnxruntime/torch provider.
    device: str = field(default_factory=lambda: _env("SCANLATION_DEVICE", "cpu"))

    # Data root (sqlite cache/TM) and model weights root (volumes in Docker).
    base_dir: Path = field(
        default_factory=lambda: Path(_env("SCANLATION_BASE_DIR", str(Path(__file__).resolve().parent.parent)))
    )

    host: str = field(default_factory=lambda: _env("SCANLATION_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("SCANLATION_PORT", "4000")))

    # Default engine selection (role -> plugin name) and languages.
    default_detector: str = field(default_factory=lambda: _env("SCANLATION_DETECTOR", "dummy"))
    default_recognizer: str = field(default_factory=lambda: _env("SCANLATION_RECOGNIZER", "dummy"))
    default_translator: str = field(default_factory=lambda: _env("SCANLATION_TRANSLATOR", "dummy"))
    default_lang_src: str = field(default_factory=lambda: _env("SCANLATION_LANG_SRC", "ja"))
    default_lang_dst: str = field(default_factory=lambda: _env("SCANLATION_LANG_DST", "ko"))

    # Manual translations win over machine ones in the TM lookup.
    favor_manual: bool = True

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def models_dir(self) -> Path:
        return Path(_env("SCANLATION_MODELS_DIR", str(self.base_dir / "models")))

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
