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

import threading
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
        # Serializes instance create+load so a key loads exactly once, even when
        # callers no longer hold an external lock. get() runs on a threadpool
        # worker (run_in_threadpool), so this is a threading.Lock, not asyncio.
        # Never hold it across anything that re-enters registry.get (non-reentrant).
        self._lock = threading.Lock()
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
    def get(self, role: str, name: str, device: str | None = None) -> Any:
        key = (role, name)
        inst = self._instances.get(key)  # lock-free fast path (atomic dict.get under the GIL)
        if inst is not None:
            return inst
        with self._lock:  # miss: serialize create+load so two threads don't double-load a key
            inst = self._instances.get(key)  # re-check: another thread may have loaded it
            if inst is None:
                inst = self._classes[role][name]()
                if device is not None:
                    # Per-engine override; honored by LocalModelEngineBase.load(),
                    # ignored by engines that don't load onto a device (translators).
                    inst._device_override = device
                inst.load()  # heavy + GIL-releasing; held under the lock on purpose (loads once)
                self._instances[key] = inst  # publish only after a successful load
            return inst

    def unload_one(self, role: str, name: str) -> None:
        """Unload + forget a single cached instance so its next get() reloads it
        (e.g. after its per-engine device override changed). No-op if not loaded.
        Call under the GPU lock so no inference is mid-flight on it."""
        with self._lock:  # symmetric with get(): no concurrent re-create mid-unload
            inst = self._instances.pop((role, name), None)
            if inst is not None:
                try:
                    inst.unload()
                except Exception:  # noqa: BLE001 - a broken unload must not block the switch
                    pass


ensure_on_path()  # volume-installed engine packages are importable before discovery
registry = Registry()
