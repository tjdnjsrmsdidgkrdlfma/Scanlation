"""A persistent worker-process pool for the recognize half of the pipeline.

The recognizer (a GPU VLM like PaddleOCR-VL) is compute-bound, and a single B=1
request leaves the GPU partly idle. Running several B=1 recognizes AT ONCE fills
that idle — the measured lever (see ``tools/recognize-gpu-speed.md``: ~1.38x at
W=4). It must be *processes*, not threads: ROCm has no MPS, so only separate
device contexts time-slice the idle in; threads on one default stream serialise.
Crop batching was the other candidate and was rejected (straggler + O(n²) vision;
see ``tools/recognize-crop-batching.md``).

So this mirrors ``tools/bench_recognize_gpu_concurrency.py`` for production: a
``ProcessPoolExecutor`` (spawn) whose workers each load the recognizer once and
hold it resident, and a page's deskewed crops fan out across them (order
preserved). The pool is a SERVER concern (like ``translate_sem``), keyed on
(engine, device, workers); the worker count is a per-engine setting resolved by
``state.resolve_recognize_concurrency`` (1 = no pool, the in-process per-crop
loop — the default, byte-identical to before).

Invariant (shared with ``_bench_common``): NOTHING heavy at import time. torch and
the engine plugin are imported only inside the worker, so importing this module in
the main process (and re-importing it in a spawned worker) stays cheap. The main
process never loads the recognizer when the pool is used — the workers own it, so
its VRAM lives once per worker, not also in the app process.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool  # not re-exported by the package top-level

logger = logging.getLogger("scanlation.recognize_pool")

# --- per-worker process globals (set once by the pool initializer) -----------
_REC = None


def _worker_init(group: str, name: str, device: str | None) -> None:
    """Runs once in each freshly spawned worker: resolve the recognizer class by
    entry-point name (the same discovery the registry uses, minus its process-wide
    state machine), load it onto the resolved device, and hold it resident in a
    process global. One model copy per worker — that copy is the VRAM cost the
    worker count is capped by. The first real recognize absorbs the kernel JIT
    (cold start); there is no synthetic warmup here (it would only move the same
    one-time cost, since the pool builds lazily on the first request anyway)."""
    global _REC
    from importlib.metadata import entry_points

    from app.plugins_path import ensure_on_path

    ensure_on_path()  # volume-installed engine packages importable in the worker too
    try:
        eps = entry_points(group=group)
    except TypeError:  # Python < 3.10 API
        eps = entry_points().get(group, [])
    cls = next((ep.load() for ep in eps if ep.name == name), None)
    if cls is None:
        raise RuntimeError(f"recognizer {name!r} not found in entry-point group {group!r}")
    rec = cls()
    if device:
        rec._device_override = device  # honored by LocalModelEngineBase.load()
    rec.load()
    _REC = rec


def _recognize_one(item) -> str:
    """One B=1 recognize in the worker. ``item`` is ``(crop, options)``; the crop
    is already deskewed upright by the caller. The recognizer's ``region`` arg is a
    throwaway (recognize reads the crop pixels, not the geometry — same as the
    bench), so it isn't shipped across the process boundary."""
    crop, options = item
    from scanlation_sdk.contracts import Region

    region = Region.from_bbox(0, 0, crop.width, crop.height)
    return _REC.recognize(crop, region, options).strip()


class RecognizePool:
    """Process-wide singleton owning the recognize worker pool. Rebuilt lazily when
    (engine, device, workers) changes; torn down on idle/engine/device/W change and
    at shutdown. SELF-PROTECTED: an in-flight-run counter guarded by a Condition lets
    teardown DRAIN live ``run``s before shutting the executor down, so the pool no
    longer leans on an external gpu_lock for that safety. Concurrent readers (the
    InferenceGate lets up to K run at once) can call ``run`` together —
    ``ProcessPoolExecutor.submit`` is thread-safe — and a teardown (ensure key-change
    / invalidate / shutdown) waits them out instead of shutting an executor mid-map."""

    def __init__(self) -> None:
        self._ex: ProcessPoolExecutor | None = None
        self._key: tuple[str, str, int] | None = None  # (name, device, workers)
        # Condition = lock + wait/notify. Guards the executor-lifecycle transitions
        # AND the in-flight counter; teardown waits on it until _inflight hits 0.
        self._cond = threading.Condition()
        self._inflight = 0

    def ensure(self, name: str, device: str | None, workers: int) -> None:
        """Build the pool for (name, device, workers) if it isn't already that. A
        change drains in-flight runs, tears the old pool down (releasing its VRAM),
        then builds new."""
        key = (name, device or "", int(workers))
        with self._cond:
            if self._ex is not None and self._key == key:
                return
            self._teardown_locked()   # waits for in-flight runs, then shuts the old ex
            self._build_locked(key)

    def run(self, items: list) -> list[str]:
        """Recognize every ``(crop, options)`` in ``items``, results aligned to input
        order. Registers as in-flight so a concurrent teardown waits for it. On
        ``BrokenProcessPool`` (a worker died/OOMed) rebuild the broken executor once
        and retry; if the retry also breaks, drop the pool (next request rebuilds
        fresh) and propagate — this request fails rather than silently loading the
        model into the main process (which would double the VRAM the pool isolates)."""
        with self._cond:
            ex, key = self._ex, self._key
            if ex is None:
                raise RuntimeError("recognize pool not built; call ensure() first")
            self._inflight += 1
        try:
            return self._map_with_retry(ex, key, items)
        finally:
            with self._cond:
                self._inflight -= 1
                if self._inflight == 0:
                    self._cond.notify_all()   # wake any teardown waiting to drain

    def invalidate(self, name: str | None = None) -> None:
        """Tear the pool down so the next ``ensure`` rebuilds it — after a device or
        worker-count change. ``name`` filters to that engine (a change to a
        non-active recognizer is then a no-op, since the pool holds only the active
        one)."""
        with self._cond:
            if self._ex is None:
                return
            if name is None or (self._key is not None and self._key[0] == name):
                self._teardown_locked()

    def shutdown(self) -> None:
        """Drain in-flight runs then terminate the workers (reclaim their VRAM).
        Called from the app lifespan finally so spawned workers don't outlive the
        server."""
        with self._cond:
            self._teardown_locked()

    # --- internals ---
    def _map_with_retry(self, ex: ProcessPoolExecutor, key, items: list) -> list[str]:
        """Map OUTSIDE the lock (long GPU work). On a broken pool, rebuild only the
        broken executor (no drain — a broken map has no live run to wait for, so this
        can't deadlock on this run's own _inflight) and retry once."""
        try:
            return list(ex.map(_recognize_one, items))
        except BrokenProcessPool:
            logger.warning("recognize pool broke (worker died/OOM); rebuilding + retrying once")
        ex2 = self._rebuild_broken(ex, key)
        try:
            return list(ex2.map(_recognize_one, items))
        except BrokenProcessPool:
            self._drop_if_current(ex2)   # retry broke too -> drop so the next request rebuilds
            raise

    def _rebuild_broken(self, ex: ProcessPoolExecutor, key) -> ProcessPoolExecutor:
        """Replace a broken executor with a fresh one and return it; if another thread
        already replaced it, return the current one. No drain (the broken ex's map has
        already errored, so there's nothing live to wait for)."""
        with self._cond:
            if self._ex is ex:
                self._shutdown_ex_locked()
                self._build_locked(key)
            return self._ex

    def _drop_if_current(self, ex: ProcessPoolExecutor) -> None:
        """After a retry also broke, drop the executor (if still current) so the next
        request rebuilds fresh."""
        with self._cond:
            if self._ex is ex:
                self._shutdown_ex_locked()

    # --- lifecycle (call under self._cond) ---
    def _build_locked(self, key: tuple[str, str, int]) -> None:
        from .registry import ROLES  # already imported at app start; no re-discovery

        name, device, workers = key
        ctx = mp.get_context("spawn")  # fork + CUDA/HIP is unsafe
        self._ex = ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, initializer=_worker_init,
            initargs=(ROLES["recognizer"], name, device or None),
        )
        self._key = key
        logger.info("recognize pool: %d workers for %r on %s", workers, name, device or "default")

    def _teardown_locked(self) -> None:
        """Drain in-flight runs, then shut the executor down. Only called from OUTSIDE
        a run's own map (ensure/invalidate/shutdown), never from _map_with_retry —
        that would wait on this run's own _inflight and deadlock."""
        while self._inflight > 0:
            self._cond.wait()
        self._shutdown_ex_locked()

    def _shutdown_ex_locked(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True)
            self._ex = None
            self._key = None


recognize_pool = RecognizePool()
