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

import _bootstrap  # noqa: F401 - side effects: add package root to sys.path, UTF-8 stdio

from app.plugins_install import find_class


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="*", help="engine names (default: ctd mangaocr)")
    args = ap.parse_args()

    for name in args.engines or ["ctd", "mangaocr"]:
        cls = find_class(name)
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
