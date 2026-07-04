"""Runtime plugin installation — pip-install a plugin's package on demand.

The core image/venv ships NO engine. Real engines (rtdetr/mangaocr/ollama/
llamacpp) are separate pip packages that live in this monorepo but are NOT in the
core image. The admin "install" button (``POST /install_plugins/``) pip-installs
the chosen one into a persistent, ``sys.path``-ed dir (``SCANLATION_PLUGINS_DIR``,
a mounted volume in Docker) — pulling the package + its heavy backend deps
(onnxruntime/torch/httpx) only then. "설치한 패키지 = 탑재 엔진" holds inside the
container too, and the install survives container recreation via the volume.

Where the package comes from:
  * default — ``pip install "scanlation-rtdetr @ git+<repo>@<ref>#subdirectory=
    packages/scanlation-rtdetr"`` (SCANLATION_ENGINE_REPO / _REF). No engine code is
    baked into the image; it's fetched from GitHub at install time.
  * dev/offline override — if ``SCANLATION_ENGINES_SRC`` points at the local
    ``packages/`` tree, install from those source dirs instead (no network).
scanlation-sdk is co-installed the same way (pip can't resolve the local/private
sdk from an index). manga-ocr steers torch to the CPU wheel index.

Two layers, kept distinct: **package** (this module, pip-install) and **weights**
(each plugin's ``install()``). ``install_plugin()`` does package-then-weights and
re-discovers entry_points live so the engine appears without a restart.

The set of *installable* engines (the catalog) is a static manifest in
``app.catalog``; this module only consumes it to drive the pip install.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import os
import queue
import site
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Callable, Iterator

from scanlation_sdk.context import context

from .catalog import CatalogEntry, catalog

DEFAULT_REPO = "https://github.com/tjdnjsrmsdidgkrdlfma/Scanlation.git"


# --- paths / repo ---------------------------------------------------------
def plugins_dir() -> Path:
    """Where plugin packages are pip-installed (a mounted volume in Docker)."""
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


def _pip_cmd(entry: CatalogEntry) -> list[str]:
    """The ``pip install --upgrade --target=<vol> …`` argv for a plugin. Shared by
    the blocking ``install_package`` and the streaming ``_stream_pip`` so the
    command (and its git+/local-source resolution) lives in one place.

    ``--upgrade`` is load-bearing for the co-installed sdk: without it, pip skips
    a package already present in the target dir, so a plugin bringing a newer sdk
    (new module/API) would keep running against the stale sdk on the /plugins
    volume and ImportError. With it, sdk is always reinstalled to match."""
    return [
        sys.executable, "-m", "pip", "install", "--upgrade",
        "--target", str(plugins_dir()),
        *entry.pip_args,
        *_install_sources(entry),
    ]


def install_package(entry: CatalogEntry) -> None:
    """pip-install the plugin's package (+ its deps) into ``plugins_dir()``.
    Raises RuntimeError with pip's stderr tail on failure."""
    plugins_dir().mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(_pip_cmd(entry), capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"pip install {entry.package} failed: {tail}")


def refresh_registry() -> None:
    """Make a just-installed package's entry_points visible without a restart."""
    from .registry import registry

    importlib.invalidate_caches()
    registry.rediscover()


def find_class(name: str):
    """The engine class registered under ``name`` in any role, or None. Shared by
    install_plugin and tools/install.py so the role-crossing lookup lives once."""
    from .registry import registry

    for mapping in registry.all_classes().values():
        if name in mapping:
            return mapping[name]
    return None


def install_plugin(name: str) -> dict:
    """Install a plugin end-to-end: pip-install its package if missing (→ live
    rediscover), then download its weights. ``name`` is the engine/registry name.
    Returns a small status dict; raises ValueError/RuntimeError on failure."""
    result: dict = {}
    cls = find_class(name)
    if cls is None:  # package not installed yet -> pip install from the catalog
        entry = catalog().get(name)
        if entry is None:
            raise ValueError(f"unknown plugin: {name}")
        install_package(entry)
        refresh_registry()
        cls = find_class(name)
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


# --- streaming install (live progress for the /admin log view) -------------
# `install_plugin` above is the plain, blocking one-shot (used by tools/install.py
# and POST /install_plugins/). The machinery below is the same install with live
# output: it runs the blocking work in a thread that pushes (kind, text) items
# onto a queue, and yields them as NDJSON-ready event dicts so /admin can show the
# pip log and the weights-download bars as they happen.

class _LineTee(io.TextIOBase):
    """A write-through sink for ``redirect_stdout/stderr`` during a plugin's
    weights download: forwards each finished segment to ``put``. Segments ended by
    ``\\r`` are ``progress`` (a live bar the UI overwrites in place); by ``\\n`` are
    ``log`` (appended). ``isatty()`` returns True so tqdm / huggingface_hub keep
    their progress bars enabled when writing to us instead of silencing them."""

    def __init__(self, put: Callable[[tuple], None]) -> None:
        self._put = put
        self._buf = ""

    def isatty(self) -> bool:  # noqa: D401 - keep tqdm bars on
        return True

    def write(self, s: str) -> int:
        self._buf += s
        while True:
            nl, cr = self._buf.find("\n"), self._buf.find("\r")
            hits = [i for i in (nl, cr) if i >= 0]
            if not hits:
                break
            i = min(hits)
            seg, delim, self._buf = self._buf[:i].strip(), self._buf[i], self._buf[i + 1:]
            if seg:
                self._put(("progress" if delim == "\r" else "log", seg))
        return len(s)

    def flush(self) -> None:  # emit any trailing partial line
        seg, self._buf = self._buf.strip(), ""
        if seg:
            self._put(("log", seg))


def _stream_pip(entry: CatalogEntry, put: Callable[[tuple], None]) -> None:
    """Run the plugin's pip install, forwarding each stdout/stderr line to ``put``
    as it arrives. Raises RuntimeError with the tail on non-zero exit (non-TTY pip
    prints plain newline-terminated lines, so no bar parsing is needed)."""
    plugins_dir().mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        _pip_cmd(entry),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    tail: deque[str] = deque(maxlen=40)
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\r\n")
        if line:
            tail.append(line)
            put(("log", line))
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"pip install {entry.package} failed: " + " | ".join(list(tail)[-6:]))


# --- in-progress tracking --------------------------------------------------
# Installs run in a background thread that outlives the streaming request, so a
# reloaded /admin must be able to tell what's still installing (to show "설치 중"
# instead of "설치") and a second click must not start a duplicate concurrent
# install of the same plugin (two pip installs into one --target corrupt it).
_installing: set[str] = set()
_installing_lock = threading.Lock()


def installing_names() -> list[str]:
    """Names whose install is running right now (surfaced by GET /get_settings/)."""
    with _installing_lock:
        return sorted(_installing)


def _begin_install(name: str) -> bool:
    """Claim ``name`` as installing; False if it was already in progress."""
    with _installing_lock:
        if name in _installing:
            return False
        _installing.add(name)
        return True


def install_plugin_events(name: str) -> Iterator[dict]:
    """Streaming variant of :func:`install_plugin`: yields progress events for the
    /admin live-log view. Event shapes (each a JSON object, one per line):

      {"event": "phase", "phase": "package" | "weights"}
      {"event": "log",   "stream": "log" | "progress", "line": "…"}
      {"event": "done",  "result": {"package": …, "weights": …}}
      {"event": "error", "message": "…"}

    Same two layers as ``install_plugin`` (pip package, then weights), done in one
    call. The blocking work runs in a daemon thread feeding a queue so lines flush
    live; this generator just drains the queue until the sentinel. Refuses (an
    ``error`` event) if the same plugin is already installing."""
    if not _begin_install(name):
        yield {"event": "error", "message": f"{name} is already installing"}
        return
    q: "queue.Queue[tuple | None]" = queue.Queue()

    def worker() -> None:
        try:
            cls = find_class(name)
            if cls is None:  # package missing -> pip install from the catalog
                entry = catalog().get(name)
                if entry is None:
                    raise ValueError(f"unknown plugin: {name}")
                q.put(("phase", "package"))
                _stream_pip(entry, q.put)
                refresh_registry()
                cls = find_class(name)
                if cls is None:
                    raise RuntimeError(f"{name} installed but not discovered (check entry_points)")
                pkg = "installed"
            else:
                pkg = "present"
            inst = cls()
            if not inst.is_installed():
                q.put(("phase", "weights"))
                sink = _LineTee(q.put)
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    inst.install()  # download weights (no-op for API-only engines)
                sink.flush()
            weights = "installed" if inst.is_installed() else "n/a"
            q.put(("done", {"package": pkg, "weights": weights}))
        except Exception as exc:  # noqa: BLE001
            q.put(("error", str(exc)))
        finally:
            with _installing_lock:
                _installing.discard(name)
            q.put(None)  # sentinel

    threading.Thread(target=worker, daemon=True, name=f"install-{name}").start()

    while True:
        item = q.get()
        if item is None:
            break
        kind, payload = item
        if kind in ("log", "progress"):
            yield {"event": "log", "stream": kind, "line": payload}
        elif kind == "phase":
            yield {"event": "phase", "phase": payload}
        elif kind == "done":
            yield {"event": "done", "result": payload}
        elif kind == "error":
            yield {"event": "error", "message": payload}
