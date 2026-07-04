"""Orchestrator internals not covered by the route tests: in-flight dedup and
the HTTP-free bad-image error. (The full pipeline path is covered end-to-end by
test_routes_run via the dummy engines.)"""
from __future__ import annotations

import asyncio

from tests.helpers import run


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
    test_decode_image_raises_bad_image,
]

if __name__ == "__main__":
    import sys

    sys.exit(run(TESTS, "test_orchestrator"))
