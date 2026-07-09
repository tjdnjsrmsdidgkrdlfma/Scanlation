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

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from scanlation_sdk.contracts import EngineBase

import app.registry as registry_mod
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


def test_device_resolver_applied_on_load():
    """get() reads the wired device_resolver at load time and stamps the override
    onto the fresh instance. Reverting get() to ignore the resolver turns this red."""
    _register_probe()
    saved = registry.device_resolver
    try:
        registry.device_resolver = lambda n: "cuda:7" if n == _PROBE_KEY[1] else None
        inst = registry.get(*_PROBE_KEY)
        assert getattr(inst, "_device_override", None) == "cuda:7"
    finally:
        registry.device_resolver = saved
        _cleanup_probe()


def test_no_device_resolver_leaves_default():
    """With no resolver wired (tools/tests), get() loads without stamping a device
    override -> the engine falls back to its DEFAULT_DEVICE, as before B3."""
    _register_probe()
    saved = registry.device_resolver
    try:
        registry.device_resolver = None
        inst = registry.get(*_PROBE_KEY)
        assert getattr(inst, "_device_override", None) is None
    finally:
        registry.device_resolver = saved
        _cleanup_probe()


class _BrokenEntryPoint:
    """A discovery entry_point whose load() fails — stands in for a ghost/stale
    install-metadata entry that importlib.metadata still enumerates (H1)."""
    name = "__broken_probe__"

    def load(self):
        raise ImportError("deliberate probe failure")


def test_discover_logs_failed_load():
    """A failing ep.load() is logged, not silently swallowed — otherwise a ghost
    entry_point vanishes with no trace and splits an engine's name across registry
    and catalog (H9). Reverting the fix (bare `except: pass`) turns this red."""
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logging.getLogger("scanlation.registry").addHandler(handler)

    real_entry_points = registry_mod.entry_points

    def fake_entry_points(*args, **kwargs):
        eps = list(real_entry_points(*args, **kwargs))
        if kwargs.get("group") == "scanlation.detectors":
            eps.append(_BrokenEntryPoint())  # inject the ghost into detectors only
        return eps

    registry_mod.entry_points = fake_entry_points
    try:
        registry.rediscover()  # re-scan with the broken entry_point present
        assert any("__broken_probe__" in r.getMessage() for r in records), \
            "failed ep.load() was not logged"
        assert not registry.has("detector", "__broken_probe__"), \
            "a failed load must not register a class"
    finally:
        registry_mod.entry_points = real_entry_points
        logging.getLogger("scanlation.registry").removeHandler(handler)
        registry.rediscover()  # rebuild a clean class map without the probe


TESTS = [
    test_concurrent_get_loads_once,
    test_cache_hit_does_not_reload,
    test_evict_then_reload,
    test_device_resolver_applied_on_load,
    test_no_device_resolver_leaves_default,
    test_discover_logs_failed_load,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_registry"))
