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
    ``python -m tests`` should use."""
    rc = 0
    for mod in modules:
        rc |= run(mod.TESTS, mod.__name__)
    return rc


# --- shared engine test bodies (the parts every plugin's suite copied) ---

def http_translator_contract(cls, success_payload: dict) -> list:
    """The two backend-agnostic HttpTranslator tests, as a list of test callables
    to splice into a plugin suite's ``TESTS``.

    ``cls`` is the translator class; ``success_payload`` is what a faked ``_post``
    returns for a successful call — its extracted text must be ``"x"`` (ollama:
    ``{"response": "x"}``, llama.cpp: ``{"choices": [{"message": {"content": "x"}}]}``).
    Both fake ``_post`` (the SDK seam), so only the payload shape differs. Backend-
    specific request/batch assertions stay in each suite."""

    def test_missing_model_raises():
        tr = cls()
        tr._post = lambda path, body: success_payload  # must never be reached
        raised = False
        try:
            tr.translate("これは十分に長い文章です", "ja", "ko", {})  # no model in options
        except ValueError:
            raised = True
        assert raised, "translate must raise when no model is selected"

    def test_blank_skips_but_short_text_translates():
        tr = cls()
        calls = {"n": 0}

        def fake(path, body):
            calls["n"] += 1
            return success_payload

        tr._post = fake
        assert tr.translate("  ", "ja", "ko", {}) == ""              # blank -> no model call
        assert calls["n"] == 0
        assert tr.translate("あ", "ja", "ko", {"model": "m"}) == "x"  # 1-char now goes to the model
        assert calls["n"] == 1

    return [test_missing_model_raises, test_blank_skips_but_short_text_translates]


def recognizer_smoke(cls, spec_module: str, missing_pkg_skip: str, missing_weights_skip: str):
    """A recognizer's weights-gated smoke, as a single test callable.

    Skips (returns the given ``SKIP:`` string) when ``spec_module`` isn't importable
    or the engine's weights aren't installed; otherwise loads the engine and asserts
    ``recognize`` returns a str for an upright white crop. ``cls`` is the recognizer
    class. Detectors have a different call/assert shape and keep their own smoke."""

    def test_recognize_returns_str():
        import importlib.util

        if importlib.util.find_spec(spec_module) is None:
            return missing_pkg_skip
        from PIL import Image

        from scanlation_sdk.contracts import Region

        rec = cls()
        if not rec.is_installed():
            return missing_weights_skip
        rec.load()
        out = rec.recognize(Image.new("RGB", (160, 64), (255, 255, 255)), Region.from_bbox(0, 0, 160, 64), {})
        assert isinstance(out, str)

    return test_recognize_returns_str
