"""Registry thread-safety — get() loads a key exactly once under concurrency.

The registry's internal lock (registry._lock) replaces the external gpu_lock as
the thing that keeps `registry.get` from double-loading an engine when two
requests first-use the same key at once. These tests drive `get` from many
threads (as the threadpool would once the gpu_lock is loosened) and assert the
model is loaded once. They register a throwaway probe engine into the live
singleton registry and MUST clean it out (finally) so later test modules that
enumerate engines never see it.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from scanlation_sdk.contracts import EngineBase

from app.registry import registry

from tests.helpers import run

_PROBE_KEY = ("detector", "__load_probe__")
_counter_lock = threading.Lock()


class _LoadProbe(EngineBase):
    """Counts load() calls across all instances. A sleep widens the race window
    so an unguarded get() would double-load (load_count >= 2)."""
    name = "__load_probe__"
    display_name = "Load probe"
    description = "Test-only: proves registry.get loads a key exactly once."

    load_count = 0  # class-level: total load() calls (one per created instance)

    def load(self) -> None:
        time.sleep(0.02)
        with _counter_lock:
            type(self).load_count += 1


def _register_probe() -> None:
    _LoadProbe.load_count = 0
    registry.all_classes()[_PROBE_KEY[0]][_PROBE_KEY[1]] = _LoadProbe


def _cleanup_probe() -> None:
    registry.unload_one(*_PROBE_KEY)  # drop the cached instance
    registry.all_classes()[_PROBE_KEY[0]].pop(_PROBE_KEY[1], None)  # drop the class map entry


def test_concurrent_get_loads_once():
    """8 threads first-use the same key simultaneously -> one load, one instance.
    Without registry._lock this fails probabilistically (load_count >= 2)."""
    _register_probe()
    try:
        n = 8
        barrier = threading.Barrier(n)

        def worker():
            barrier.wait()  # release all threads into get() at the same instant
            return registry.get(*_PROBE_KEY)

        with ThreadPoolExecutor(max_workers=n) as pool:
            results = [f.result() for f in [pool.submit(worker) for _ in range(n)]]

        assert _LoadProbe.load_count == 1, f"loaded {_LoadProbe.load_count}x, expected 1"
        assert len({id(r) for r in results}) == 1, "threads got different instances"
    finally:
        _cleanup_probe()


def test_cache_hit_does_not_reload():
    """A second get() returns the cached instance without reloading."""
    _register_probe()
    try:
        a = registry.get(*_PROBE_KEY)
        b = registry.get(*_PROBE_KEY)
        assert a is b
        assert _LoadProbe.load_count == 1  # hit path never calls load() again
    finally:
        _cleanup_probe()


def test_evict_then_reload():
    """unload_one() drops the instance so the next get() reloads a fresh one."""
    _register_probe()
    try:
        a = registry.get(*_PROBE_KEY)
        registry.unload_one(*_PROBE_KEY)
        b = registry.get(*_PROBE_KEY)
        assert a is not b
        assert _LoadProbe.load_count == 2  # evict forces exactly one reload
    finally:
        _cleanup_probe()


TESTS = [
    test_concurrent_get_loads_once,
    test_cache_hit_does_not_reload,
    test_evict_then_reload,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_registry"))
