"""RecognizePool self-protection: teardown drains in-flight runs, concurrent runs
share the pool, and a broken-pool rebuild doesn't deadlock on its own in-flight
counter. A fake executor stands in for the real spawn ProcessPoolExecutor (no worker
processes) so the locking is exercised without a GPU — the real multiprocess run is
bench-validated separately.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures.process import BrokenProcessPool

from app.recognize_pool import RecognizePool

from tests.helpers import run


class _FakeExecutor:
    """Stand-in for ProcessPoolExecutor. ``map`` echoes OUT-<i> per item (optionally
    blocking on an Event so a test can hold a run 'in-flight'); ``shutdown`` counts."""
    def __init__(self, block: threading.Event | None = None):
        self.block = block
        self.shutdown_calls = 0
        self.map_started = threading.Event()

    def map(self, fn, items):
        self.map_started.set()
        if self.block is not None:
            self.block.wait(2.0)
        return [f"OUT-{i}" for i in range(len(list(items)))]

    def shutdown(self, wait=True):
        self.shutdown_calls += 1


def _mk_pool(fake: _FakeExecutor, key=("eng", "", 2)) -> RecognizePool:
    pool = RecognizePool()
    pool._ex = fake
    pool._key = key
    return pool


def test_single_run_returns_ordered():
    """A lone run maps every item and returns results in input order; inflight nets 0."""
    pool = _mk_pool(_FakeExecutor())
    assert pool.run([("a", {}), ("b", {})]) == ["OUT-0", "OUT-1"]
    assert pool._inflight == 0


def test_teardown_waits_for_inflight_run():
    """invalidate() must DRAIN an in-flight run (not shut the executor mid-map): it
    blocks until the run finishes, then shuts down exactly once."""
    block = threading.Event()
    fake = _FakeExecutor(block=block)
    pool = _mk_pool(fake)

    out: dict = {}
    tr = threading.Thread(target=lambda: out.setdefault("r", pool.run([("crop", {})])))
    tr.start()
    assert fake.map_started.wait(2.0)          # run is inside ex.map, blocked

    td = threading.Thread(target=pool.invalidate)
    td.start()
    time.sleep(0.1)
    assert fake.shutdown_calls == 0            # teardown is draining, NOT shut down yet
    assert pool._inflight == 1

    block.set()                                 # let the run finish
    tr.join(2.0); td.join(2.0)
    assert fake.shutdown_calls == 1             # teardown completed after the drain
    assert pool._ex is None
    assert out["r"] == ["OUT-0"]


def test_concurrent_runs_share_pool():
    """Two runs use the same executor at once (inflight reaches 2) with no teardown."""
    block = threading.Event()
    fake = _FakeExecutor(block=block)
    pool = _mk_pool(fake)

    threads = [threading.Thread(target=lambda: pool.run([("crop", {})])) for _ in range(2)]
    for t in threads:
        t.start()
    deadline = time.time() + 2.0
    while time.time() < deadline and pool._inflight < 2:
        time.sleep(0.005)
    assert pool._inflight == 2                   # both runs share the pool concurrently
    assert fake.shutdown_calls == 0
    block.set()
    for t in threads:
        t.join(2.0)
    assert pool._inflight == 0


def test_broken_pool_rebuild_no_selfdeadlock():
    """A broken map rebuilds the executor and retries — the rebuild must NOT drain its
    own in-flight run (that would deadlock); the retry then succeeds on the new pool."""
    broken = _FakeExecutor()
    broken.map = lambda fn, items: (_ for _ in ()).throw(BrokenProcessPool("boom"))
    good = _FakeExecutor()
    pool = _mk_pool(broken)
    # _rebuild_broken calls _build_locked; stand in for spawning fresh workers.
    pool._build_locked = lambda key: (setattr(pool, "_ex", good), setattr(pool, "_key", key))

    out = pool.run([("crop", {})])              # broken -> rebuild -> retry on good
    assert out == ["OUT-0"]
    assert pool._ex is good
    assert broken.shutdown_calls == 1           # the broken executor was shut down
    assert pool._inflight == 0


def test_broken_retry_also_breaks_drops_and_raises():
    """If the retry also breaks, the pool is dropped (next request rebuilds) and the
    BrokenProcessPool propagates rather than falling back to an in-process load."""
    b1 = _FakeExecutor(); b1.map = lambda fn, items: (_ for _ in ()).throw(BrokenProcessPool("boom1"))
    b2 = _FakeExecutor(); b2.map = lambda fn, items: (_ for _ in ()).throw(BrokenProcessPool("boom2"))
    pool = _mk_pool(b1)
    pool._build_locked = lambda key: (setattr(pool, "_ex", b2), setattr(pool, "_key", key))

    raised = False
    try:
        pool.run([("crop", {})])
    except BrokenProcessPool:
        raised = True
    assert raised
    assert pool._ex is None                     # dropped so the next request rebuilds fresh
    assert pool._inflight == 0


TESTS = [
    test_single_run_returns_ordered,
    test_teardown_waits_for_inflight_run,
    test_concurrent_runs_share_pool,
    test_broken_pool_rebuild_no_selfdeadlock,
    test_broken_retry_also_breaks_drops_and_raises,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_recognize_pool"))
