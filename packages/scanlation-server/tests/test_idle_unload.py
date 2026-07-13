"""Idle-unload sweep: a local engine unused past the /admin window is dropped from
VRAM; the window at 0 disables the sweep; a recently-used engine survives; and
translators (HTTP/out-of-process) are never touched.

sweep_once takes a monotonic ``now`` so these tests are deterministic — they load a
probe, backdate its registry._last_used, and assert what the sweep does. gpu_gate is
reset to a fresh (unbound) InferenceGate per run so each asyncio.run() loop can bind
it without tripping asyncio's bound-to-a-different-loop guard.
"""
from __future__ import annotations

import asyncio

from scanlation_sdk.contracts import EngineBase

from app.idle_unload import sweep_once
from app.registry import registry
from app.state import InferenceGate, state

from tests.helpers import run

_DET_KEY = ("detector", "__idle_probe__")
_TR_KEY = ("translator", "__idle_tr_probe__")


class _IdleProbe(EngineBase):
    name = "__idle_probe__"
    display_name = "Idle probe"
    description = "Test-only: idle-unload sweep target."
    loaded = 0
    unloaded = 0

    def load(self) -> None:
        type(self).loaded += 1

    def unload(self) -> None:
        type(self).unloaded += 1


class _TrProbe(_IdleProbe):
    name = "__idle_tr_probe__"
    display_name = "Idle translator probe"
    description = "Test-only: translator (HTTP) — must never be idle-unloaded."


def _register(role: str, name: str, cls: type) -> None:
    cls.loaded = cls.unloaded = 0
    registry.all_classes()[role][name] = cls


def _cleanup(role: str, name: str) -> None:
    registry.unload_one(role, name)
    registry.all_classes()[role].pop(name, None)


def _run(coro):
    # Fresh unbound gate so this call's loop binds it; hand the next consumer a clean
    # one on the way out (asyncio primitives bind to the first loop that awaits them).
    state.gpu_gate = InferenceGate(1)
    try:
        return asyncio.run(coro)
    finally:
        state.gpu_gate = InferenceGate(1)


def test_sweep_unloads_idle_local_engine():
    """A detector unused past the window is unloaded; the sweep returns its key and
    the registry forgets both the instance and its last-used timestamp."""
    _register(*_DET_KEY, _IdleProbe)
    saved = state.selection.model_idle_unload_minutes
    try:
        state.selection.model_idle_unload_minutes = 5   # ttl 300s
        registry.get(*_DET_KEY)                          # load + cache + bump last_used
        registry._last_used[_DET_KEY] = 0.0             # pretend it's ancient
        unloaded = _run(sweep_once(10_000.0))            # now well past the deadline
        assert _DET_KEY in unloaded
        assert _IdleProbe.unloaded == 1
        assert _DET_KEY not in registry._instances       # dropped from the cache
        assert _DET_KEY not in registry._last_used       # unload_one clears the timestamp
    finally:
        state.selection.model_idle_unload_minutes = saved
        _cleanup(*_DET_KEY)


def test_sweep_disabled_when_zero():
    """model_idle_unload_minutes == 0 -> never unload, even a long-idle engine."""
    _register(*_DET_KEY, _IdleProbe)
    saved = state.selection.model_idle_unload_minutes
    try:
        state.selection.model_idle_unload_minutes = 0
        registry.get(*_DET_KEY)
        registry._last_used[_DET_KEY] = 0.0
        unloaded = _run(sweep_once(10_000.0))
        assert unloaded == []
        assert _IdleProbe.unloaded == 0
        assert _DET_KEY in registry._instances
    finally:
        state.selection.model_idle_unload_minutes = saved
        _cleanup(*_DET_KEY)


def test_sweep_skips_recently_used():
    """A local engine used more recently than the window survives the sweep."""
    _register(*_DET_KEY, _IdleProbe)
    saved = state.selection.model_idle_unload_minutes
    try:
        state.selection.model_idle_unload_minutes = 5   # ttl 300s
        registry.get(*_DET_KEY)
        registry._last_used[_DET_KEY] = 9_950.0         # 50s ago at now=10_000, < 300
        unloaded = _run(sweep_once(10_000.0))
        assert unloaded == []
        assert _IdleProbe.unloaded == 0
        assert _DET_KEY in registry._instances
    finally:
        state.selection.model_idle_unload_minutes = saved
        _cleanup(*_DET_KEY)


def test_sweep_never_touches_translator():
    """Translators hold no VRAM here (HTTP/out-of-process) -> excluded even if idle."""
    _register(*_TR_KEY, _TrProbe)
    saved = state.selection.model_idle_unload_minutes
    try:
        state.selection.model_idle_unload_minutes = 5
        registry.get(*_TR_KEY)
        registry._last_used[_TR_KEY] = 0.0
        unloaded = _run(sweep_once(10_000.0))
        assert unloaded == []
        assert _TrProbe.unloaded == 0
        assert _TR_KEY in registry._instances
    finally:
        state.selection.model_idle_unload_minutes = saved
        _cleanup(*_TR_KEY)


TESTS = [
    test_sweep_unloads_idle_local_engine,
    test_sweep_disabled_when_zero,
    test_sweep_skips_recently_used,
    test_sweep_never_touches_translator,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_idle_unload"))
