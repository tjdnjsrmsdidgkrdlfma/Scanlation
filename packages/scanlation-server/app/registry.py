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

import logging
import threading
import time
from typing import Any, Callable

from .plugins_path import ensure_on_path, iter_entry_points

logger = logging.getLogger("scanlation.registry")

ROLES: dict[str, str] = {
    "detector": "scanlation.detectors",
    "recognizer": "scanlation.recognizers",
    "translator": "scanlation.translators",
}
# The three role ids in fixed order — the single source everything else derives.
ROLE_NAMES: tuple[str, ...] = tuple(ROLES)
# Roles whose engines load in-process onto a compute device (torch/VRAM). Only
# these are worth reclaiming on idle; translators are out-of-process HTTP clients
# (ollama's own OLLAMA_KEEP_ALIVE governs the LLM's residency). See state.Selection.
LOCAL_ROLES: tuple[str, ...] = ("detector", "recognizer")


class Registry:
    def __init__(self) -> None:
        self._classes: dict[str, dict[str, type]] = {r: {} for r in ROLES}
        self._instances: dict[tuple[str, str], Any] = {}
        # key -> time.monotonic() of last get(), for the idle-unload sweep. Bumped on
        # every resolve (hit + load); dropped on unload. A single-key dict write is
        # atomic under the GIL, so the fast path bumps it lock-free like the hit above.
        self._last_used: dict[tuple[str, str], float] = {}
        # Per-engine compute-device resolver, wired at the composition root (main
        # lifespan) so the registry needn't import state. A device is a LOAD-TIME
        # property: it's read only when a key is first loaded (below), never on a
        # cache hit — changing an engine's device means unload_one, then reload.
        # Left None in tools/tests -> engines load on their DEFAULT_DEVICE.
        self.device_resolver: Callable[[str], str | None] | None = None
        # Serializes instance create+load so a key loads exactly once, even when
        # callers no longer hold an external lock. get() runs on a threadpool
        # worker (run_in_threadpool), so this is a threading.Lock, not asyncio.
        # Never hold it across anything that re-enters registry.get (non-reentrant).
        self._lock = threading.Lock()
        self._discover()

    def _discover(self) -> None:
        for role, group in ROLES.items():
            for ep in iter_entry_points(group):
                try:
                    self._classes[role][ep.name] = ep.load()
                except Exception as exc:  # noqa: BLE001 - a broken/absent engine must not kill discovery
                    # ...but leave a trace. A silently-swallowed load failure hides
                    # a ghost entry_point (stale install metadata) that then splits
                    # an engine's name across registry and catalog with no symptom.
                    logger.warning("skipping %s entry_point %r: failed to load (%r)", role, ep.name, exc)

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
        inst = self._instances.get(key)  # lock-free fast path (atomic dict.get under the GIL)
        if inst is not None:
            self._last_used[key] = time.monotonic()  # bump for the idle sweep (lock-free)
            return inst
        with self._lock:  # miss: serialize create+load so two threads don't double-load a key
            inst = self._instances.get(key)  # re-check: another thread may have loaded it
            if inst is None:
                inst = self._classes[role][name]()
                device = self.device_resolver(name) if self.device_resolver else None
                if device is not None:
                    # Per-engine override; honored by LocalModelEngineBase.load(),
                    # ignored by engines that don't load onto a device (translators).
                    inst._device_override = device
                inst.load()  # heavy + GIL-releasing; held under the lock on purpose (loads once)
                self._instances[key] = inst  # publish only after a successful load
            self._last_used[key] = time.monotonic()  # freshly loaded / re-check hit: mark used
            return inst

    def unload_one(self, role: str, name: str) -> None:
        """Unload + forget a single cached instance so its next get() reloads it
        (e.g. after its per-engine device override changed). No-op if not loaded.
        Call under the GPU lock so no inference is mid-flight on it."""
        with self._lock:  # symmetric with get(): no concurrent re-create mid-unload
            self._last_used.pop((role, name), None)
            inst = self._instances.pop((role, name), None)
            if inst is not None:
                try:
                    inst.unload()
                except Exception:  # noqa: BLE001 - a broken unload must not block the switch
                    pass

    def idle_candidates(self, ttl_seconds: float, now: float) -> list[tuple[str, str]]:
        """Loaded LOCAL engines (detector/recognizer — the in-process torch/VRAM
        roles) not used for more than ``ttl_seconds`` as of ``now`` (a monotonic
        clock). The idle-unload sweep reads this lock-free to pick candidates, then
        re-checks each under the GPU lock before unloading (a request may have just
        bumped it). Translators are excluded — they hold no VRAM here."""
        return [
            key for key in list(self._instances)
            if key[0] in LOCAL_ROLES
            and (last := self._last_used.get(key)) is not None
            and now - last > ttl_seconds
        ]


ensure_on_path()  # volume-installed engine packages are importable before discovery
registry = Registry()
