"""Runtime plugin (engine) installation — pip-install an engine package on demand.

The core image/venv ships only ``dummy``. Real engines (ctd/mangaocr/ollama/
llamacpp) are separate pip packages whose *source* sits unbuilt under
``SCANLATION_ENGINES_SRC`` (Docker: ``/opt/engines``; bare-metal: the repo
``packages/``), with **zero dependency on the core until installed**. The admin
"install" button (``POST /manage_plugins/``) calls here to
``pip install --target=$SCANLATION_PLUGINS_DIR <source>`` — pulling the package
plus its heavy backend deps (onnxruntime/torch/httpx) into a persistent,
``sys.path``-ed dir. In Docker that dir is a mounted volume, so installed engines
survive container recreation; "설치한 패키지 = 탑재 엔진" holds inside the container too.

Two layers, kept distinct:
  * **package** (this module) — pip-install the plugin code + backend libs.
  * **weights** (each plugin's ``install()``) — download model files.
``install_engine()`` does package-then-weights; the registry is re-discovered
live after a package lands so it appears without a restart.

Catalog: each engine source dir is discovered by reading its ``pyproject.toml``
(no import of the not-yet-installed package). ``name``/``description`` come from
``[project]``; the engine name(s) and role(s) come from the
``scanlation.<role>`` entry-point groups it declares.
"""
from __future__ import annotations

import importlib
import os
import site
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from scanlation_sdk.context import context

# entry-point group (in pyproject) -> our role name.
_GROUP_TO_ROLE = {
    "scanlation.detectors": "detector",
    "scanlation.recognizers": "recognizer",
    "scanlation.translators": "translator",
}

# Engine source dirs to ignore in the catalog: the core + sdk are not installable
# engines (dummy already ships in the core). Only relevant to the bare-metal
# fallback where SCANLATION_ENGINES_SRC = the whole `packages/` tree.
_SKIP_PACKAGES = {"scanlation-sdk", "scanlation-server"}

# Per-package extra pip args. manga-ocr pulls torch; the default PyPI linux wheel
# is the multi-GB CUDA build, so steer it to the CPU index (its `+cpu` local
# version outranks the plain wheel, so pip prefers it).
_EXTRA_PIP_ARGS: dict[str, list[str]] = {
    "scanlation-mangaocr": ["--extra-index-url", "https://download.pytorch.org/whl/cpu"],
}


@dataclass
class CatalogEntry:
    name: str                       # engine name = registry key (e.g. "ctd")
    package: str                    # pip/dist name (e.g. "scanlation-ctd")
    source: Path                    # dir containing pyproject.toml
    description: str = ""
    roles: list[str] = field(default_factory=list)
    pip_args: list[str] = field(default_factory=list)


# --- paths ----------------------------------------------------------------
def plugins_dir() -> Path:
    """Where engine packages are pip-installed (a mounted volume in Docker)."""
    env = os.environ.get("SCANLATION_PLUGINS_DIR")
    return Path(env) if env else context.base_dir / "plugins"


def engines_src() -> Path:
    """Root holding engine *source* dirs. Docker sets SCANLATION_ENGINES_SRC=
    /opt/engines; bare-metal falls back to the repo `packages/` (this file lives
    at packages/scanlation-server/app/plugins_install.py, so parents[2]=packages)."""
    env = os.environ.get("SCANLATION_ENGINES_SRC")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


def sdk_src() -> Path | None:
    """The ``scanlation-sdk`` source dir, co-installed with each engine so pip can
    satisfy the engine's ``scanlation-sdk`` dep from a local path (it's not on any
    index). SCANLATION_SDK_SRC overrides; else it sits beside the engine sources."""
    env = os.environ.get("SCANLATION_SDK_SRC")
    if env:
        p = Path(env)
        return p if p.is_dir() else None
    cand = engines_src() / "scanlation-sdk"
    return cand if cand.is_dir() else None


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


# --- catalog (installable engines, read without importing them) -----------
def catalog() -> dict[str, CatalogEntry]:
    """Discover installable engines by parsing each source dir's pyproject.toml.
    Keyed by engine name (the entry-point name, e.g. "ctd")."""
    out: dict[str, CatalogEntry] = {}
    root = engines_src()
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        pp = d / "pyproject.toml"
        if not pp.is_file():
            continue
        try:
            data = tomllib.loads(pp.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        proj = data.get("project", {})
        pkg = proj.get("name", d.name)
        if pkg in _SKIP_PACKAGES:
            continue
        eps = proj.get("entry-points") or {}
        for group, mapping in eps.items():
            role = _GROUP_TO_ROLE.get(group)
            if not role or not isinstance(mapping, dict):
                continue
            for epname in mapping:
                entry = out.get(epname)
                if entry is None:
                    out[epname] = CatalogEntry(
                        name=epname,
                        package=pkg,
                        source=d,
                        description=proj.get("description", ""),
                        roles=[role],
                        pip_args=_EXTRA_PIP_ARGS.get(pkg, []),
                    )
                elif role not in entry.roles:
                    entry.roles.append(role)
    return out


# --- install --------------------------------------------------------------
def install_package(entry: CatalogEntry) -> None:
    """pip-install the engine package (+ its deps) into ``plugins_dir()``.
    Raises RuntimeError with pip's stderr tail on failure."""
    target = plugins_dir()
    target.mkdir(parents=True, exist_ok=True)
    sources = []
    sdk = sdk_src()
    if sdk is not None:  # co-install so the engine's scanlation-sdk dep resolves locally
        sources.append(str(sdk))
    sources.append(str(entry.source))
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--target", str(target),
        *entry.pip_args,
        *sources,
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
