"""Thin entry point for the compare_models research harness; see compare/cli.py."""
import _bootstrap  # noqa: F401 - side effects: add package root to sys.path, UTF-8 stdio
from compare.cli import main

if __name__ == "__main__":
    main()
