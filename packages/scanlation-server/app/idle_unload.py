"""Background idle-unload of local model engines.

The local (torch) engines — detector and recognizer — live in worker PROCESSES,
outside the registry (see engine_pool), and hold their model + GPU context for as
long as those workers run. Between reading sessions that means VRAM stays taken and
the GPU can't runtime-suspend. This module runs a periodic sweep that tears a pool
down once it's gone unused for the /admin-configured window
(state.selection.model_idle_unload_minutes; 0 = never) — the analog of ollama's
OLLAMA_KEEP_ALIVE for the LLM. Killing the workers is what frees the VRAM AND drops
their HIP context, letting the GPU reach D3cold (~0W); an in-process engine could
never do the second part (the server process pins its context for life).

The registry sweep stays for any engine that a tool or test loaded in-process; in
the server itself the pools are the path that matters.

Wired in the FastAPI lifespan (app.main): sweep_loop() runs as a task started at
startup and cancelled at shutdown.
"""
from __future__ import annotations

import asyncio
import logging
import time

from starlette.concurrency import run_in_threadpool

from .engine_pool import POOLS
from .registry import registry
from .state import state

logger = logging.getLogger("scanlation.idle_unload")

# How often the sweep wakes. This is the GRANULARITY of idle detection (an engine
# unloads within one interval of its idle deadline), not a user-facing knob, so it
# stays a module constant — like the other internal cadence/format constants the
# project deliberately keeps out of /admin.
_SWEEP_INTERVAL_SECONDS = 30


async def sweep_once(now: float) -> list[tuple[str, str]]:
    """One pass: unload every local engine idle past the configured window as of
    ``now`` (a monotonic clock). No-op when the window is 0 (disabled) or nothing is
    idle. Re-checks idleness under the GPU lock so a request that arrived while we
    waited for the lock isn't unloaded out from under itself (get() bumps last-used
    to a monotonic >= now, so such a key drops out of the second idle_candidates).
    Returns the keys unloaded, for logging/tests."""
    minutes = state.selection.model_idle_unload_minutes
    if minutes <= 0:
        return []
    ttl = minutes * 60
    pool_due = any((idle := p.idle_seconds(now)) is not None and idle >= ttl for p in POOLS)
    if not registry.idle_candidates(ttl, now) and not pool_due:  # cheap lock-free early-out
        return []
    # writer() excludes in-flight RECOGNIZE; each pool's own drain excludes an in-flight
    # run of ITS role (detect runs off the gate, so nothing else could).
    async with state.gpu_gate.writer():
        return await run_in_threadpool(_unload_idle_locked, ttl, now, minutes)


def _unload_idle_locked(ttl: float, now: float, minutes: int) -> list[tuple[str, str]]:
    """Re-check idleness and unload. Runs in a threadpool: a pool teardown waits for an
    in-flight run to finish (a detect is ~270ms) and must not block the event loop."""
    unloaded: list[tuple[str, str]] = []
    for role, name in registry.idle_candidates(ttl, now):  # re-check (a request may have bumped it)
        registry.unload_one(role, name)
        unloaded.append((role, name))
        logger.info("idle-unloaded %s %r (unused > %dm)", role, name, minutes)
    # The pools' models live in their worker PROCESSES, not the registry, so the loop
    # above never sees them. Tear idle workers down — this is what lets their GPU reach
    # D3cold (~0W); an in-process engine can't (the server process pins its context for
    # life). invalidate() drains any in-flight run of that role first.
    for pool in POOLS:
        idle = pool.idle_seconds(now)
        if idle is not None and idle >= ttl:
            pool.invalidate()
            unloaded.append((pool.role, "__pool__"))
            logger.info("idle-unloaded %s pool (unused > %dm)", pool.role, minutes)
    return unloaded


async def sweep_loop() -> None:
    """Wake every _SWEEP_INTERVAL_SECONDS and run one sweep. Runs until cancelled
    (lifespan shutdown). A single failed pass is logged and the loop continues — a
    transient unload error must not kill idle reclaim for the process's lifetime."""
    while True:
        await asyncio.sleep(_SWEEP_INTERVAL_SECONDS)
        try:
            await sweep_once(time.monotonic())
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a bad pass must not kill the loop
            logger.warning("idle-unload sweep failed", exc_info=True)
