"""Plugin-facing runtime context: the slice of config engine plugins need —
where model weights live (``models_dir``) and the language tables. Read from the
same env vars the server core uses (``SCANLATION_MODELS_DIR`` /
``SCANLATION_BASE_DIR``), so this is the single source of truth for both core and
plugins — no drift. Self-contained: never imports the server package. Per-engine
compute device is not here — it lives in each engine's DEFAULT_DEVICE + admin
override (there is no global device).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# iso1 -> human readable. The small useful set the popup language dropdowns show.
LANGUAGES: dict[str, str] = {
    "ja": "Japanese",
    "ko": "Korean",
    "en": "English",
    "zh": "Chinese",
}

# iso1 -> plain language name passed to translators (ollama wants "japanese").
LANG_PLAIN: dict[str, str] = {k: v.lower() for k, v in LANGUAGES.items()}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Context:
    """Filesystem config shared by the core and every engine plugin.

    ``base_dir`` defaults to the current working directory (the server is
    launched from its package dir; Docker/tests set it via env). Per-engine
    compute device is not here — see the module docstring.
    """

    base_dir: Path = field(
        default_factory=lambda: Path(_env("SCANLATION_BASE_DIR", str(Path.cwd())))
    )

    @property
    def models_dir(self) -> Path:
        env = os.environ.get("SCANLATION_MODELS_DIR")
        return Path(env) if env else self.base_dir / "models"


context = Context()
