"""Persistent worker-process pools for the local (torch) halves of the pipeline.

Local engines run in worker PROCESSES, never in the server process. Two reasons,
and both are load-bearing:

* **Throughput (recognize).** The recognizer (a GPU VLM like PaddleOCR-VL) is
  compute-bound, and a single B=1 request leaves the GPU partly idle. Running
  several B=1 recognizes AT ONCE fills that idle — the measured lever (see
  ``tools/recognize-gpu-speed.md``: ~1.38x at W=4). It must be *processes*, not
  threads: ROCm has no MPS, so only separate device contexts time-slice the idle
  in; threads on one default stream serialise. Crop batching was the other
  candidate and was rejected (straggler + O(n²) vision; see
  ``tools/recognize-crop-batching.md``).
* **Idle power (both roles).** A torch/HIP context opens the GPU's kfd + render
  nodes and holds them for the PROCESS's lifetime — pinning every card at D0, so
  the GPU can never runtime-suspend (~0W) between reading sessions. Third-party
  code reaches for the GPU whether or not the engine wants one: transformers
  probes ``torch.cuda.device_count()`` while choosing an attention implementation
  and again inside the forward, so even a CPU-only detector pins the cards when it
  loads in-process. Env masking does not help (ROCm's amdsmi enumerates every
  physical GPU BEFORE ``HIP_VISIBLE_DEVICES`` is applied). Only a process that
  EXITS releases the context — hence workers, torn down by the idle-unload sweep.
  The server core must never touch torch.cuda (see also ``gpus.list_gpus``, which
  probes in a throwaway subprocess for the same reason).

So this mirrors ``tools/bench_recognize_gpu_concurrency.py`` for production: a
``ProcessPoolExecutor`` (spawn) whose workers each load one engine once and hold
it resident, and work fans out across them (order preserved). A pool is a SERVER
concern (like ``translate_sem``), keyed on (engine, device, workers).

One class, two singletons — ``recognize_pool`` (workers = the per-engine
``state.resolve_recognize_concurrency``; a page's crops fan out) and
``detect_pool`` (always 1 worker: detect is once per page, so there is nothing to
fan out — the pool is there for process isolation, not parallelism).

Invariant (shared with ``_bench_common``): NOTHING heavy at import time. torch and
the engine plugin are imported only inside the worker, so importing this module in
the main process (and re-importing it in a spawned worker) stays cheap.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool  # not re-exported by the package top-level

logger = logging.getLogger("scanlation.engine_pool")

# --- per-worker process globals (set once by the pool initializer) -----------
_ENGINE = None

# --- TEMP occupancy bench (revert via git) -----------------------------------
# Every worker crop reports its wall-clock execution window (time.time(), which IS
# comparable across the spawned worker processes) into this main-process sink, so a
# controlled batch can measure how full the W-worker pool actually was. This is the
# direct test of the "gate+K is an image bundle -> workers idle when a page's crops
# sum < W" claim: utilization ~1.0 means the pool was never starved for crops (image
# boundaries cost ~nothing -> stage separation can't help); <1.0 is the worker
# capacity lost to starvation. Reset before a batch, read after (see bench routes).
_OCC_LOCK = threading.Lock()
_OCC: list[tuple[float, float]] = []


def _worker_init(group: str, name: str, device: str | None) -> None:
    """Runs once in each freshly spawned worker: resolve the engine class by
    entry-point name (the same discovery the registry uses, minus its process-wide
    state machine), load it onto the resolved device, and hold it resident in a
    process global. One model copy per worker — that copy is the VRAM cost the
    worker count is capped by. The first real call absorbs the kernel JIT (cold
    start); there is no synthetic warmup here (it would only move the same one-time
    cost, since the pool builds lazily on the first request anyway)."""
    global _ENGINE
    from app.plugins_path import ensure_on_path, iter_entry_points

    ensure_on_path()  # volume-installed engine packages importable in the worker too
    cls = next((ep.load() for ep in iter_entry_points(group) if ep.name == name), None)
    if cls is None:
        raise RuntimeError(f"engine {name!r} not found in entry-point group {group!r}")
    eng = cls()
    if device:
        eng._device_override = device  # honored by LocalModelEngineBase.load()
    eng.load()
    _ENGINE = eng


def _recognize_one(item) -> tuple[str, float, float, float]:
    """One B=1 recognize in the worker. ``item`` is ``(crop, options)``; the crop is
    already deskewed upright by the caller. Returns ``(text, elapsed_ms, t_start,
    t_end)`` — ``elapsed_ms`` (perf_counter) is the per-crop recognize stat; the two
    ``time.time()`` wall stamps are the execution window for the TEMP occupancy bench
    (perf_counter isn't comparable across processes; time.time() is). The recognizer's
    ``region`` arg is a throwaway (recognize reads the crop pixels, not the geometry —
    same as the bench), so it isn't shipped across the process boundary."""
    crop, options = item
    from scanlation_sdk.contracts import Region

    region = Region.from_bbox(0, 0, crop.width, crop.height)
    t_start = time.time()
    t0 = time.perf_counter()
    text = _ENGINE.recognize(crop, region, options).strip()
    ms = (time.perf_counter() - t0) * 1000
    return text, ms, t_start, time.time()


def _detect_one(item) -> list:
    """One page's detect in the worker. ``item`` is ``(image, options)``; returns the
    raw ``list[Region]``. Reading order is assigned by the CALLER (pipeline), not here
    — it's pure geometry with no model, so it stays in the main process where the rest
    of the page's bookkeeping lives."""
    image, options = item
    return _ENGINE.detect(image, options)


# --- TEMP occupancy bench helpers (revert via git) ---------------------------
def _split_occ(raw: list) -> list:
    """Peel ``(t_start, t_end)`` off the 4-tuple recognize results into the occupancy
    sink and return the ``(text, ms)`` pairs the pipeline expects. Anything else passes
    through untouched — detect results (a ``list[Region]``) and the unit-test fake
    executor's placeholder strings — so this instrumentation is invisible to them."""
    out, occ = [], []
    for r in raw:
        if isinstance(r, tuple) and len(r) == 4:
            text, ms, ts, te = r
            out.append((text, ms))
            occ.append((ts, te))
        else:
            out.append(r)
    if occ:
        with _OCC_LOCK:
            _OCC.extend(occ)
    return out


def reset_occupancy() -> None:
    """Clear the sink before a measured batch."""
    with _OCC_LOCK:
        _OCC.clear()


def active_workers() -> int:
    """W of the live recognize pool (0 if none built) — the denominator for utilization."""
    key = recognize_pool._key
    return key[2] if key else 0


def occupancy_stats(workers: int) -> dict:
    """From the collected per-crop wall execution windows, how full the W-worker pool
    was during the batch. Sweep-line over interval starts/ends gives the time-weighted
    distribution of concurrently-executing workers; utilization = avg_busy / W. Robust
    to GPU time-slicing (a worker holding a slow time-sliced crop still counts as busy
    — only crop STARVATION shows as idle, which is exactly the claim under test)."""
    with _OCC_LOCK:
        intervals = [iv for iv in _OCC if iv[1] > iv[0]]
    if not intervals:
        return {"crops": 0, "workers": workers}
    start = min(s for s, _ in intervals)
    end = max(e for _, e in intervals)
    wall = end - start
    busy = sum(e - s for s, e in intervals)
    events = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals])
    hist: dict[int, float] = {}
    cur, prev = 0, start
    for t, d in events:
        hist[cur] = hist.get(cur, 0.0) + (t - prev)
        cur += d
        prev = t
    avg = busy / wall if wall else 0.0
    return {
        "crops": len(intervals),
        "workers": workers,
        "wall_s": round(wall, 3),
        "busy_worker_s": round(busy, 3),
        "avg_busy_workers": round(avg, 3),
        "utilization": round(avg / workers, 4) if workers else None,
        "pct_wall_by_busy_count": {str(c): round(100 * s / wall, 1)
                                   for c, s in sorted(hist.items())},
    }


class EnginePool:
    """Process-wide owner of ONE role's worker pool. Rebuilt lazily when (engine,
    device, workers) changes; torn down on idle/engine/device/W change and at
    shutdown. SELF-PROTECTED: an in-flight-run counter guarded by a Condition lets
    teardown DRAIN live ``run``s before shutting the executor down — which is also
    what makes a separate caller-side lock unnecessary (an /admin device change or an
    idle-unload can't tear an engine out mid-forward). Concurrent readers (the
    InferenceGate lets up to K run at once) can call ``run`` together —
    ``ProcessPoolExecutor.submit`` is thread-safe — and a teardown (ensure key-change
    / invalidate / shutdown) waits them out instead of shutting an executor mid-map.

    ``role`` selects the entry-point group its workers resolve engines in;
    ``task`` is the module-level function each worker runs per item (it must be
    picklable, so it lives at module scope, not on this class)."""

    def __init__(self, role: str, task) -> None:
        self.role = role
        self._task = task
        self._ex: ProcessPoolExecutor | None = None
        self._key: tuple[str, str, int] | None = None  # (name, device, workers)
        # Condition = lock + wait/notify. Guards the executor-lifecycle transitions
        # AND the in-flight counter; teardown waits on it until _inflight hits 0.
        self._cond = threading.Condition()
        self._inflight = 0
        # time.monotonic() of the last run() (and of the initial build), or None when no
        # pool is live. The idle-unload sweep reads this via idle_seconds() to tear the
        # workers down after the /admin window — freeing their VRAM AND letting the GPU
        # drop to D3cold (~0W), which an in-process engine can't (the server process pins
        # its HIP context for life).
        self._last_used: float | None = None

    def ensure(self, name: str, device: str | None, workers: int = 1) -> None:
        """Build the pool for (name, device, workers) if it isn't already that. A
        change drains in-flight runs, tears the old pool down (releasing its VRAM),
        then builds new."""
        key = (name, device or "", int(workers))
        with self._cond:
            if self._ex is not None and self._key == key:
                return
            self._teardown_locked()   # waits for in-flight runs, then shuts the old ex
            self._build_locked(key)

    def run(self, items: list) -> list:
        """Run the role's task over every item, results aligned to input order.
        Registers as in-flight so a concurrent teardown waits for it. On
        ``BrokenProcessPool`` (a worker died/OOMed) rebuild the broken executor once
        and retry; if the retry also breaks, drop the pool (next request rebuilds
        fresh) and propagate — this request fails rather than silently loading the
        model into the main process (which would double the VRAM the pool isolates)."""
        with self._cond:
            ex, key = self._ex, self._key
            if ex is None:
                raise RuntimeError(f"{self.role} pool not built; call ensure() first")
            self._inflight += 1
        try:
            return self._map_with_retry(ex, key, items)
        finally:
            with self._cond:
                self._inflight -= 1
                self._last_used = time.monotonic()   # idle clock resets on each run
                if self._inflight == 0:
                    self._cond.notify_all()   # wake any teardown waiting to drain

    def invalidate(self, name: str | None = None) -> None:
        """Tear the pool down so the next ``ensure`` rebuilds it — after a device or
        worker-count change. ``name`` filters to that engine (a change to a
        non-active engine is then a no-op, since the pool holds only the active
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

    def idle_seconds(self, now: float) -> float | None:
        """Monotonic seconds since the last run (``now`` from time.monotonic()), or None
        when no pool is live. The idle-unload sweep uses this to tear the workers down
        after the /admin window: the engine lives in the worker processes, not the
        registry, so the registry-based sweep can't see it — this is the hook that does."""
        with self._cond:
            if self._ex is None or self._last_used is None:
                return None
            return now - self._last_used

    # --- internals ---
    def _map_with_retry(self, ex: ProcessPoolExecutor, key, items: list) -> list:
        """Map OUTSIDE the lock (long model work). On a broken pool, rebuild only the
        broken executor (no drain — a broken map has no live run to wait for, so this
        can't deadlock on this run's own _inflight) and retry once."""
        try:
            return _split_occ(list(ex.map(self._task, items)))
        except BrokenProcessPool:
            logger.warning("%s pool broke (worker died/OOM); rebuilding + retrying once", self.role)
        ex2 = self._rebuild_broken(ex, key)
        try:
            return _split_occ(list(ex2.map(self._task, items)))
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
            initargs=(ROLES[self.role], name, device or None),
        )
        self._key = key
        self._last_used = time.monotonic()   # start the idle clock at build (first run bumps it)
        logger.info("%s pool: %d workers for %r on %s", self.role, workers, name, device or "default")

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
            self._last_used = None


recognize_pool = EnginePool("recognizer", _recognize_one)
detect_pool = EnginePool("detector", _detect_one)

# Both pools, for the callers that treat them uniformly (idle-unload sweep, lifespan
# shutdown, /admin device change) — so adding a third role touches one list, not three.
POOLS: tuple[EnginePool, ...] = (detect_pool, recognize_pool)
