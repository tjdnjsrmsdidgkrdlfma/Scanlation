"""Where runtime-installed plugin packages live, and making that dir importable.

Split out of ``plugins_install`` so ``registry`` can put the volume on ``sys.path``
before its first entry_points scan without importing the installer — which imports
``registry`` right back. A leaf module: it knows the path, nothing about pip.
"""
from __future__ import annotations

import os
import site
import sys
from importlib.metadata import entry_points
from pathlib import Path

from scanlation_sdk.context import context


def iter_entry_points(group: str) -> list:
    """Every entry point declared for ``group``, across the stdlib API gap: 3.10+
    takes ``group=``, the 3.9 API returns a dict keyed by group. Both the registry
    (main process) and the recognize worker pool (spawned workers) discover engines
    this way, so it lives here in the leaf module they already import."""
    try:
        return list(entry_points(group=group))
    except TypeError:  # Python < 3.10 API
        return list(entry_points().get(group, []))


def plugins_dir() -> Path:
    """Where plugin packages are pip-installed (a mounted volume in Docker)."""
    env = os.environ.get("SCANLATION_PLUGINS_DIR")
    return Path(env) if env else context.base_dir / "plugins"


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
