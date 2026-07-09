"""catalog <-> plugin description drift guard.

catalog.py carries a ``description`` for each installable plugin so /admin can
show it BEFORE the package is installed (the plugin's plugin.py isn't importable
yet, so the class attribute can't be read). Once a plugin IS installed the same
string also lives on its engine class — two hand-written copies that can silently
drift.

This asserts they match for every catalog entry whose engine is currently
discovered (installed). Entries whose package isn't installed here are skipped —
so on a dev box with only some plugins installed it guards those, and on a full
install (or CI with all installed) it guards all of them.
"""
from __future__ import annotations

from app.catalog import catalog
from app.registry import registry


def _engine_class(entry):
    """The discovered engine class for a catalog entry, or None if not installed."""
    for role in entry.roles:
        if registry.has(role, entry.name):
            return registry.get_class(role, entry.name)
    return None


def test_catalog_description_matches_installed_plugin():
    checked = []
    for name, entry in catalog().items():
        cls = _engine_class(entry)
        if cls is None:
            continue  # package not installed here -> nothing to compare against
        assert entry.description == cls.description, (
            f"{name}: catalog description drifted from plugin.py\n"
            f"  catalog: {entry.description!r}\n"
            f"  plugin:  {cls.description!r}"
        )
        checked.append(name)
    if not checked:
        return "SKIP: no catalog plugins installed to compare against"


TESTS = [test_catalog_description_matches_installed_plugin]

if __name__ == "__main__":
    import sys

    from tests.helpers import run

    sys.exit(run(TESTS, "test_catalog"))
