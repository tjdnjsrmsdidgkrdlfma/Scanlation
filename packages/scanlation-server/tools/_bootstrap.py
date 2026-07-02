"""Shared CLI prologue for the tools/ scripts. Import it first, for side effects:

    import _bootstrap  # noqa: F401 - makes `app` importable + forces UTF-8 stdio

It puts the server package root (this file's grandparent) on sys.path so `app`
resolves when a tool is run as `python tools/<script>.py`, and reconfigures the
console to UTF-8 so Japanese/Korean output survives a cp949 terminal.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
