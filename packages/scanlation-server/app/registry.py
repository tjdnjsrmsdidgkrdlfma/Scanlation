"""Engine discovery + lazy instantiation.

Engines are found purely through ``importlib.metadata`` entry_points. The core
ships NO engine of its own; every engine is a separate pip package declaring the
``scanlation.<role>`` groups (auto-discovered on install). There is no hardcoded
engine map — installing a package is how an engine appears. (Tests register their
own fakes directly; see tests/fake_engines.py.)

Engines are instantiated lazily on first use — that's when VRAM/model weights
are actually loaded. Class-level metadata (OPTION_SCHEMA, description, ...) is
read without instantiating, so the handshake/options routes stay light.
"""
from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from .plugins_install import ensure_on_path

ROLES: dict[str, str] = {
    "detector": "scanlation.detectors",
    "recognizer": "scanlation.recognizers",
    "translator": "scanlation.translators",
}
# The three role ids in fixed order — the single source everything else derives.
ROLE_NAMES: tuple[str, ...] = tuple(ROLES)


class Registry:
    def __init__(self) -> None:
        self._classes: dict[str, dict[str, type]] = {r: {} for r in ROLES}
        self._instances: dict[tuple[str, str], Any] = {}
        self._discover()

    def _discover(self) -> None:
        for role, group in ROLES.items():
            try:
                eps = entry_points(group=group)
            except TypeError:  # Python < 3.10 API
                eps = entry_points().get(group, [])
            for ep in eps:
                try:
                    self._classes[role][ep.name] = ep.load()
                except Exception:  # noqa: BLE001 - a broken/absent engine must not kill discovery
                    pass

    def rediscover(self) -> None:
        """Re-scan entry_points after a package is pip-installed at runtime (the
        admin plugin installer). Clears the class map and rebuilds it; already
        loaded instances are kept so an in-use engine isn't disturbed."""
        self._classes = {r: {} for r in ROLES}
        self._discover()

    # --- queries (no instantiation) ---
    def names(self, role: str) -> list[str]:
        return sorted(self._classes[role])

    def has(self, role: str, name: str) -> bool:
        return name in self._classes[role]

    def get_class(self, role: str, name: str) -> type:
        return self._classes[role][name]

    def all_classes(self) -> dict[str, dict[str, type]]:
        return self._classes

    # --- lazy instance (loads weights on first use) ---
    def get(self, role: str, name: str) -> Any:
        key = (role, name)
        if key not in self._instances:
            inst = self._classes[role][name]()
            inst.load()
            self._instances[key] = inst
        return self._instances[key]


ensure_on_path()  # volume-installed engine packages are importable before discovery
registry = Registry()
