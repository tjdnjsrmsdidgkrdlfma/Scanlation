"""app/gpus.py — host GPU enumeration for the device picker.

Torch is stubbed so these run deterministically on a GPU-less host. list_gpus is
lru_cached, so every test clears the cache before and after so a stub can't leak
into a later real call.
"""
from __future__ import annotations

import sys
import types

from app.gpus import list_gpus

from tests.helpers import run


def _fake_torch(count):
    """Minimal torch stub exposing the cuda bits list_gpus reads."""
    m = types.ModuleType("torch")
    m.cuda = types.SimpleNamespace(
        is_available=lambda: count > 0,
        device_count=lambda: count,
        get_device_name=lambda i: f"GPU-{i}",
    )
    return m


def _with_torch(fake):
    """Run list_gpus() with sys.modules['torch'] swapped for `fake` (None -> the
    import fails), cache cleared around it, original restored."""
    orig = sys.modules.get("torch")
    list_gpus.cache_clear()
    try:
        sys.modules["torch"] = fake  # None -> `import torch` raises -> []
        return list_gpus()
    finally:
        list_gpus.cache_clear()
        if orig is not None:
            sys.modules["torch"] = orig
        else:
            sys.modules.pop("torch", None)


def test_list_gpus_no_torch():
    assert _with_torch(None) == []


def test_list_gpus_no_gpu():
    assert _with_torch(_fake_torch(0)) == []


def test_list_gpus_enumerates():
    assert _with_torch(_fake_torch(2)) == [
        {"index": 0, "name": "GPU-0"},
        {"index": 1, "name": "GPU-1"},
    ]


TESTS = [
    test_list_gpus_no_torch,
    test_list_gpus_no_gpu,
    test_list_gpus_enumerates,
]

if __name__ == "__main__":
    sys.exit(run(TESTS, "test_gpus"))
