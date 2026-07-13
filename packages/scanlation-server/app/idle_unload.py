"""Background idle-unload of local model engines.

Local torch engines (detector/recognizer) load onto the GPU on first use, and the
registry then holds them until the process exits or an /admin device change drops
them (registry.unload_one). Between reading sessions that means VRAM stays pinned.
This module runs a periodic sweep that drops a local engine once it's gone unused
for the /admin-configured window (state.selection.model_idle_unload_minutes; 0 =
never) — the in-process analog of ollama's OLLAMA_KEEP_ALIVE for the LLM.

Wired in the FastAPI lifespan (app.main): sweep_loop() runs as a task started at
startup and cancelled at shutdown.
"""
from __future__ import annotations

import asyncio
import logging
import time

from starlette.concurrency import run_in_threadpool

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
    if not registry.idle_candidates(ttl, now):  # cheap lock-free early-out
        return []
    # writer() excludes in-flight RECOGNIZE; the threadpool'd helper also holds
    # detect_lock to exclude an in-flight DETECT (which runs off the gate now).
    async with state.gpu_gate.writer():
        return await run_in_threadpool(_unload_idle_locked, ttl, now, minutes)


def _unload_idle_locked(ttl: float, now: float, minutes: int) -> list[tuple[str, str]]:
    """Re-check idleness and unload under detect_lock — so an in-flight DETECT isn't
    dropped mid-forward (get() bumped its last-used to a monotonic >= now, so a key
    used while we waited for the lock drops out of this second idle_candidates).
    Runs in a threadpool: acquiring detect_lock may wait on a detect (~270ms) and must
    not block the event loop."""
    unloaded: list[tuple[str, str]] = []
    with state.detect_lock:
        for role, name in registry.idle_candidates(ttl, now):  # re-check under the lock
            registry.unload_one(role, name)
            unloaded.append((role, name))
            logger.info("idle-unloaded %s %r (unused > %dm)", role, name, minutes)
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
