"""Where runtime-installed plugin packages live, and making that dir importable.

Split out of ``plugins_install`` so ``registry`` can put the volume on ``sys.path``
before its first entry_points scan without importing the installer — which imports
``registry`` right back. A leaf module: it knows the path, nothing about pip.
"""
from __future__ import annotations

import os
import site
import sys
from pathlib import Path

from scanlation_sdk.context import context


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
