"""app/gpus.py — host GPU enumeration for the device picker.

Torch is stubbed so these run deterministically on a GPU-less host. list_gpus is
lru_cached, so every test clears the cache before and after so a stub can't leak
into a later real call.
"""
from __future__ import annotations

import sys
import types

from app import gpus
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


def _detect_with(kfd, nvidia):
    """detect_gpu_vendor with /dev/kfd + /dev/nvidia* existence stubbed."""
    orig_exists, orig_glob = gpus.os.path.exists, gpus.glob.glob
    try:
        gpus.os.path.exists = lambda p: kfd if p == "/dev/kfd" else orig_exists(p)
        gpus.glob.glob = lambda pat: (["/dev/nvidia0"] if nvidia and "nvidia" in pat else [])
        return gpus.detect_gpu_vendor()
    finally:
        gpus.os.path.exists, gpus.glob.glob = orig_exists, orig_glob


def test_detect_gpu_vendor():
    assert _detect_with(kfd=True, nvidia=False) == "amd"
    assert _detect_with(kfd=False, nvidia=True) == "nvidia"
    assert _detect_with(kfd=True, nvidia=True) == "both"
    assert _detect_with(kfd=False, nvidia=False) is None


def _torch_build_with(hip, cuda):
    """installed_torch_build with a stubbed torch exposing version.hip/cuda."""
    orig = sys.modules.get("torch")
    try:
        m = types.ModuleType("torch")
        m.version = types.SimpleNamespace(hip=hip, cuda=cuda)
        sys.modules["torch"] = m
        return gpus.installed_torch_build()
    finally:
        if orig is not None:
            sys.modules["torch"] = orig
        else:
            sys.modules.pop("torch", None)


def test_installed_torch_build():
    assert _torch_build_with(hip="6.2", cuda=None) == "rocm"
    assert _torch_build_with(hip=None, cuda="12.4") == "cuda"
    assert _torch_build_with(hip=None, cuda=None) == "cpu"
    # no torch installed -> None
    orig = sys.modules.get("torch")
    try:
        sys.modules["torch"] = None   # `import torch` raises -> None
        assert gpus.installed_torch_build() is None
    finally:
        if orig is not None:
            sys.modules["torch"] = orig
        else:
            sys.modules.pop("torch", None)


TESTS = [
    test_list_gpus_no_torch,
    test_list_gpus_no_gpu,
    test_list_gpus_enumerates,
    test_detect_gpu_vendor,
    test_installed_torch_build,
]

if __name__ == "__main__":
    sys.exit(run(TESTS, "test_gpus"))
