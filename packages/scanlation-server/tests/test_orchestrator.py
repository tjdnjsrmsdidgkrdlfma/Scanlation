"""Orchestrator internals not covered by the route tests: in-flight dedup and
the HTTP-free bad-image error. (The full pipeline path is covered end-to-end by
test_routes_run via the dummy engines.)"""
from __future__ import annotations

import asyncio
import gc

from tests.helpers import run


def _asyncio_complaints(coro_factory):
    """Run ``coro_factory()`` on a fresh loop and return whatever asyncio handed its
    exception handler — including the ``__del__``-time complaint a Future whose
    exception nobody read produces. The gc.collect() is what triggers that finalizer."""
    seen: list[str] = []

    async def main():
        asyncio.get_running_loop().set_exception_handler(
            lambda _loop, ctx: seen.append(ctx.get("message", ""))
        )
        await coro_factory()
        gc.collect()
        await asyncio.sleep(0)  # let any scheduled handler run

    asyncio.run(main())
    return seen


def test_run_deduped_shares_one_computation():
    """Two concurrent calls for one key run compute once and share the result."""
    from app.orchestrator import _run_deduped

    calls = {"n": 0}

    async def compute():
        calls["n"] += 1
        await asyncio.sleep(0)  # yield so the second call finds the in-flight future
        return "R"

    async def main():
        return await asyncio.gather(
            _run_deduped(("dedup-key",), compute),
            _run_deduped(("dedup-key",), compute),
        )

    results = asyncio.run(main())
    assert results == ["R", "R"]
    assert calls["n"] == 1


def test_run_deduped_shares_one_failure_with_its_waiters():
    """A failing compute runs once; the first caller and every waiter see the SAME
    exception object, and the in-flight entry is released either way."""
    from app.orchestrator import _run_deduped
    from app.state import state

    calls = {"n": 0}
    boom = RuntimeError("engine exploded")

    async def compute():
        calls["n"] += 1
        await asyncio.sleep(0)  # yield so the second call finds the in-flight future
        raise boom

    async def main():
        return await asyncio.gather(
            _run_deduped(("fail-key",), compute),
            _run_deduped(("fail-key",), compute),
            return_exceptions=True,
        )

    got = asyncio.run(main())
    assert calls["n"] == 1                       # the failure was computed once
    assert got[0] is boom and got[1] is boom     # ...and shared, not re-raised anew
    assert ("fail-key",) not in state.inflight   # released for the next request


def test_run_deduped_solo_failure_leaves_no_unretrieved_future():
    """A failure with NO waiter must not leave an exception nobody read on the
    in-flight future: asyncio complains 'Future exception was never retrieved' when
    such a future is finalized, spraying a traceback into the log on every failed run."""
    from app.orchestrator import _run_deduped

    async def compute():
        await asyncio.sleep(0)
        raise RuntimeError("engine exploded")

    async def once():
        try:
            await _run_deduped(("solo-key",), compute)
        except RuntimeError:
            pass  # the caller handles it -- that IS the retrieval

    assert _asyncio_complaints(once) == []


def test_decode_image_raises_bad_image():
    """A non-image payload raises BadImageError, not an HTTPException (the route
    owns the 400 mapping)."""
    from app.orchestrator import BadImageError, _decode_image

    try:
        _decode_image("notanimage")  # valid base64 chars, not an image
        raise AssertionError("expected BadImageError")
    except BadImageError:
        pass


TESTS = [
    test_run_deduped_shares_one_computation,
    test_run_deduped_shares_one_failure_with_its_waiters,
    test_run_deduped_solo_failure_leaves_no_unretrieved_future,
    test_decode_image_raises_bad_image,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_orchestrator"))
