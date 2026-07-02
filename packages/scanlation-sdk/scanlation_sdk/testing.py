"""Hand-rolled test runner shared across packages (no pytest).

Each test file defines plain ``test_*`` functions and a ``TESTS`` list, then
``run(TESTS, title)`` executes them (AssertionError -> FAILED, other -> ERROR,
a returned "SKIP..." string -> skipped) and returns 0 (all ok/skipped) or 1.
The server core and every engine package reuse this so their ``python -m tests``
report identically without a pytest dependency.
"""
from __future__ import annotations


def run(tests, title: str) -> int:
    """Run zero-arg test callables; print O/X/- per test; return 0 (all ok) or 1.
    A test returning a string starting with 'SKIP' is reported as skipped."""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    results: dict[str, str] = {}
    for test in tests:
        name = test.__name__
        try:
            r = test()
            results[name] = r if isinstance(r, str) and r.startswith("SKIP") else "PASSED"
        except AssertionError as e:
            results[name] = f"FAILED: {e}"
        except Exception as e:  # noqa: BLE001
            results[name] = f"ERROR: {type(e).__name__}: {e}"
    for name, res in results.items():
        status = "O" if res == "PASSED" else ("-" if res.startswith("SKIP") else "X")
        print(f"  {status} {name}: {res}")
    return 0 if all(r == "PASSED" or r.startswith("SKIP") for r in results.values()) else 1


def run_modules(modules) -> int:
    """Run every module's ``TESTS`` list (each test module defines ``TESTS``).
    Returns 0 if all modules passed/skipped, else 1 — the exit code a package's
    ``python -m tests`` should use. Replaces the identical loop each package had."""
    rc = 0
    for mod in modules:
        rc |= run(mod.TESTS, mod.__name__)
    return rc
