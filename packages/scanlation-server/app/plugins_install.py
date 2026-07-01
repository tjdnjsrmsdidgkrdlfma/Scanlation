"""Runtime plugin (engine) installation — pip-install an engine package on demand.

The core image/venv ships only ``dummy``. Real engines (ctd/mangaocr/ollama/
llamacpp) are separate pip packages that live in this monorepo but are NOT in the
core image. The admin "install" button (``POST /manage_plugins/``) pip-installs
the chosen one into a persistent, ``sys.path``-ed dir (``SCANLATION_PLUGINS_DIR``,
a mounted volume in Docker) — pulling the package + its heavy backend deps
(onnxruntime/torch/httpx) only then. "설치한 패키지 = 탑재 엔진" holds inside the
container too, and the install survives container recreation via the volume.

Where the package comes from:
  * default — ``pip install "scanlation-ctd @ git+<repo>@<ref>#subdirectory=
    packages/scanlation-ctd"`` (SCANLATION_ENGINE_REPO / _REF). No engine code is
    baked into the image; it's fetched from GitHub at install time.
  * dev/offline override — if ``SCANLATION_ENGINES_SRC`` points at the local
    ``packages/`` tree, install from those source dirs instead (no network).
scanlation-sdk is co-installed the same way (pip can't resolve the local/private
sdk from an index). manga-ocr steers torch to the CPU wheel index.

Two layers, kept distinct: **package** (this module, pip-install) and **weights**
(each plugin's ``install()``). ``install_engine()`` does package-then-weights and
re-discovers entry_points live so the engine appears without a restart.

Catalog: the set of *installable* engines is a small static manifest here — it
can't come from entry_points (those only list *installed* engines) nor from the
source (the image has none). Installed engines are still discovered purely via
entry_points in the registry; this manifest only drives the install UI.
"""
from __future__ import annotations

import importlib
import os
import site
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from scanlation_sdk.context import context

DEFAULT_REPO = "https://github.com/tjdnjsrmsdidgkrdlfma/Scanlation.git"

# The installable engines. name = registry/engine name; package = pip/dist name
# (and the packages/<package> subdir). Installed engines are found via
# entry_points; this only lists what /admin can offer to install.
_CATALOG: dict[str, dict] = {
    "ctd": {
        "package": "scanlation-ctd",
        "display_name": "comic-text-detector",
        "roles": ["detector"],
        "description": "comic-text-detector (ONNX) text-region detector.",
        "pip_args": [],
    },
    "mangaocr": {
        "package": "scanlation-mangaocr",
        "display_name": "Manga OCR",
        "roles": ["recognizer"],
        "description": "manga-ocr Japanese recognizer (needs torch — CPU wheel).",
        # steer torch to the CPU index (its +cpu local version outranks the plain
        # PyPI CUDA wheel, so pip prefers it).
        "pip_args": ["--extra-index-url", "https://download.pytorch.org/whl/cpu"],
    },
    "ollama": {
        "package": "scanlation-ollama",
        "display_name": "Ollama",
        "roles": ["translator"],
        "description": "LLM translation via a local ollama server.",
        "pip_args": [],
    },
    "llamacpp": {
        "package": "scanlation-llamacpp",
        "display_name": "llama.cpp",
        "roles": ["translator"],
        "description": "LLM translation via an OpenAI-compatible /v1 server.",
        "pip_args": [],
    },
}


@dataclass
class CatalogEntry:
    name: str                       # engine name = registry key (e.g. "ctd")
    package: str                    # pip/dist name (e.g. "scanlation-ctd")
    display_name: str = ""          # human-readable name shown before install
    description: str = ""
    roles: list[str] = field(default_factory=list)
    pip_args: list[str] = field(default_factory=list)


# --- paths / repo ---------------------------------------------------------
def plugins_dir() -> Path:
    """Where engine packages are pip-installed (a mounted volume in Docker)."""
    env = os.environ.get("SCANLATION_PLUGINS_DIR")
    return Path(env) if env else context.base_dir / "plugins"


def engine_repo() -> str:
    return os.environ.get("SCANLATION_ENGINE_REPO", DEFAULT_REPO)


def engine_ref() -> str:
    return os.environ.get("SCANLATION_ENGINE_REF", "main")


def ensure_on_path() -> None:
    """Put ``plugins_dir()`` on sys.path so already-installed (persisted) engine
    packages are importable + entry_points-discoverable. Called at registry import
    (earliest) so the first discovery already sees volume-installed engines."""
    d = plugins_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    p = str(d)
    if p not in sys.path:
        site.addsitedir(p)  # appends to sys.path (+ processes any .pth)


# --- catalog (installable engines) ----------------------------------------
def catalog() -> dict[str, CatalogEntry]:
    """The static manifest of installable engines, keyed by engine name."""
    return {
        name: CatalogEntry(
            name=name,
            package=spec["package"],
            display_name=spec.get("display_name") or name,
            description=spec["description"],
            roles=list(spec["roles"]),
            pip_args=list(spec["pip_args"]),
        )
        for name, spec in _CATALOG.items()
    }


# --- install --------------------------------------------------------------
def _install_sources(entry: CatalogEntry) -> list[str]:
    """pip requirement strings for [sdk, engine]. Local source dirs when
    SCANLATION_ENGINES_SRC points at the ``packages/`` tree (dev/offline); else
    git+ from the repo (the Docker default — no engine code baked into the image).
    sdk is co-installed so its (local/private) dep resolves without an index."""
    src = os.environ.get("SCANLATION_ENGINES_SRC")
    if src:
        base = Path(src)
        eng, sdk = base / entry.package, base / "scanlation-sdk"
        if eng.is_dir() and sdk.is_dir():
            return [str(sdk), str(eng)]
    g = f"git+{engine_repo()}@{engine_ref()}"
    return [
        f"scanlation-sdk @ {g}#subdirectory=packages/scanlation-sdk",
        f"{entry.package} @ {g}#subdirectory=packages/{entry.package}",
    ]


def install_package(entry: CatalogEntry) -> None:
    """pip-install the engine package (+ its deps) into ``plugins_dir()``.
    Raises RuntimeError with pip's stderr tail on failure."""
    target = plugins_dir()
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--target", str(target),
        *entry.pip_args,
        *_install_sources(entry),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"pip install {entry.package} failed: {tail}")


def refresh_registry() -> None:
    """Make a just-installed package's entry_points visible without a restart."""
    from .registry import registry

    importlib.invalidate_caches()
    registry.rediscover()


def _find_class(name: str):
    from .registry import registry

    for mapping in registry.all_classes().values():
        if name in mapping:
            return mapping[name]
    return None


def install_engine(name: str) -> dict:
    """Install an engine end-to-end: pip-install its package if missing (→ live
    rediscover), then download its weights. ``name`` is the engine/registry name.
    Returns a small status dict; raises ValueError/RuntimeError on failure."""
    result: dict = {}
    cls = _find_class(name)
    if cls is None:  # package not installed yet -> pip install from the catalog
        entry = catalog().get(name)
        if entry is None:
            raise ValueError(f"unknown plugin: {name}")
        install_package(entry)
        refresh_registry()
        cls = _find_class(name)
        if cls is None:
            raise RuntimeError(f"{name} installed but not discovered (check entry_points)")
        result["package"] = "installed"
    else:
        result["package"] = "present"

    inst = cls()
    if not inst.is_installed():
        inst.install()  # download weights (no-op for engines without assets)
    result["weights"] = "installed" if inst.is_installed() else "n/a"
    return result
