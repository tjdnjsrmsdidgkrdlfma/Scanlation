"""InferenceGate: bounded-concurrency reader/writer gate (no GPU, no server).

readers = the detect+recognize half (up to K at once = cross-image overlap); a
writer = a lifecycle mutation that drains all K permits for exclusivity. Each test
builds its own gate inside asyncio.run so the gate's Semaphore/Lock bind to that
call's loop (asyncio primitives bind to the first loop that awaits them).
"""
from __future__ import annotations

import asyncio

from app.state import InferenceGate

from tests.helpers import run


def test_k_readers_overlap():
    """K readers enter concurrently: with K=4, all four are inside the gate at once."""
    async def body():
        gate = InferenceGate(4)
        inside = 0
        peak = 0
        entered_all = asyncio.Event()
        release = asyncio.Event()
        n = 4

        async def reader():
            nonlocal inside, peak
            async with gate.reader():
                inside += 1
                peak = max(peak, inside)
                if inside == n:
                    entered_all.set()
                await release.wait()
                inside -= 1

        tasks = [asyncio.create_task(reader()) for _ in range(n)]
        await asyncio.wait_for(entered_all.wait(), timeout=2.0)  # all 4 in at once
        assert peak == 4
        release.set()
        await asyncio.gather(*tasks)

    asyncio.run(body())


def test_k1_is_mutex():
    """K=1: a second reader cannot enter until the first exits (single mutex)."""
    async def body():
        gate = InferenceGate(1)
        first_in = asyncio.Event()
        second_in = asyncio.Event()
        release = asyncio.Event()

        async def first():
            async with gate.reader():
                first_in.set()
                await release.wait()

        async def second():
            async with gate.reader():
                second_in.set()

        t1 = asyncio.create_task(first())
        await first_in.wait()
        t2 = asyncio.create_task(second())
        try:
            await asyncio.wait_for(second_in.wait(), timeout=0.2)
            raise AssertionError("second reader entered while first held the mutex")
        except asyncio.TimeoutError:
            pass
        release.set()
        await asyncio.gather(t1, t2)
        assert second_in.is_set()

    asyncio.run(body())


def test_writer_drains_in_flight_readers():
    """A writer waits for in-flight readers to finish before entering."""
    async def body():
        gate = InferenceGate(4)
        reader_in = asyncio.Event()
        reader_release = asyncio.Event()
        writer_in = asyncio.Event()

        async def reader():
            async with gate.reader():
                reader_in.set()
                await reader_release.wait()

        async def writer():
            async with gate.writer():
                writer_in.set()

        tr = asyncio.create_task(reader())
        await reader_in.wait()
        tw = asyncio.create_task(writer())
        try:
            await asyncio.wait_for(writer_in.wait(), timeout=0.2)
            raise AssertionError("writer entered while a reader was still in-flight")
        except asyncio.TimeoutError:
            pass
        reader_release.set()
        await asyncio.wait_for(writer_in.wait(), timeout=2.0)  # drained -> now it enters
        await asyncio.gather(tr, tw)

    asyncio.run(body())


def test_new_reader_blocks_under_writer():
    """While a writer holds the gate, a new reader cannot enter (exclusivity)."""
    async def body():
        gate = InferenceGate(2)
        writer_in = asyncio.Event()
        writer_release = asyncio.Event()
        reader_in = asyncio.Event()

        async def writer():
            async with gate.writer():
                writer_in.set()
                await writer_release.wait()

        async def reader():
            async with gate.reader():
                reader_in.set()

        tw = asyncio.create_task(writer())
        await writer_in.wait()
        tr = asyncio.create_task(reader())
        try:
            await asyncio.wait_for(reader_in.wait(), timeout=0.2)
            raise AssertionError("reader entered while a writer held the gate")
        except asyncio.TimeoutError:
            pass
        writer_release.set()
        await asyncio.wait_for(reader_in.wait(), timeout=2.0)
        await asyncio.gather(tw, tr)

    asyncio.run(body())


def test_writers_never_overlap():
    """Concurrent writers run one at a time — no deadlock draining partial permits."""
    async def body():
        gate = InferenceGate(3)
        active = 0
        peak = 0

        async def writer():
            nonlocal active, peak
            async with gate.writer():
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)  # hold long enough for overlap to show
                active -= 1

        await asyncio.gather(*[writer() for _ in range(3)])
        assert peak == 1  # writers are mutually exclusive; completing at all = no deadlock

    asyncio.run(body())


TESTS = [
    test_k_readers_overlap,
    test_k1_is_mutex,
    test_writer_drains_in_flight_readers,
    test_new_reader_blocks_under_writer,
    test_writers_never_overlap,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_inference_gate"))
