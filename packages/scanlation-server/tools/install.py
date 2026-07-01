"""Install engine resources (download weights) — the explicit, non-magic step.

    python tools/install.py                 # ctd + mangaocr (the downloadable ones)
    python tools/install.py ctd             # just CTD
    python tools/install.py ctd mangaocr

Equivalent to POST /manage_plugins/ {"plugins": {"<name>": true}} (the popup's
one-click install). load() never downloads implicitly, so run this once first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

from app.registry import registry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="*", help="engine names (default: ctd mangaocr)")
    args = ap.parse_args()

    for name in args.engines or ["ctd", "mangaocr"]:
        cls = next(
            (m[name] for m in registry.all_classes().values() if name in m), None
        )
        if cls is None:
            print(f"{name}: unknown engine", file=sys.stderr)
            continue
        inst = cls()
        if inst.is_installed():
            print(f"{name}: already installed")
            continue
        print(f"{name}: installing ...")
        inst.install()
        print(f"{name}: installed={inst.is_installed()}")


if __name__ == "__main__":
    main()
