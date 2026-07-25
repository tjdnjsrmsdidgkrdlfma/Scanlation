"""app/gpus.py — host GPU enumeration for the device picker.

list_gpus() enumerates in a THROWAWAY subprocess so the long-lived server process
never holds a HIP context (that would pin the GPU at D0 and block idle
runtime-suspend). Two layers are tested: the probe SCRIPT (app.gpus._GPU_PROBE) is
run for real against a fake ``torch`` module on a temp PYTHONPATH — exercising the
exact label logic that ships — and the list_gpus WRAPPER's stdout parsing + failure
handling is tested with subprocess.run stubbed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

from app import gpus
from app.gpus import list_gpus

from tests.helpers import run


# --- the probe script, run for real against a stub torch ---------------------
# Each fake is written to a temp dir put first on PYTHONPATH, so `import torch` in
# the child resolves to it (shadowing any real torch in the venv/plugins).

_FAKE_TORCH_2GPU = """
class _Props:
    def __init__(self, i):
        self.name = "GPU-%d" % i
        self.gcnArchName = ["gfx906:sramecc+:xnack-", "gfx1200"][i]
        self.total_memory = [32, 16][i] * 1024 ** 3
class _Cuda:
    @staticmethod
    def is_available(): return True
    @staticmethod
    def device_count(): return 2
    @staticmethod
    def get_device_properties(i): return _Props(i)
cuda = _Cuda()
"""

_FAKE_TORCH_NO_GPU = """
class _Cuda:
    @staticmethod
    def is_available(): return False
    @staticmethod
    def device_count(): return 0
cuda = _Cuda()
"""

_FAKE_TORCH_IMPORT_FAILS = 'raise ImportError("stubbed: no torch")\n'


def _run_probe(fake_torch_src):
    """Execute the REAL _GPU_PROBE in a child whose `import torch` resolves to a
    stub (temp dir prepended via PYTHONPATH, so it shadows any real torch).
    Returns the parsed JSON the probe prints — the same value list_gpus() reads."""
    d = tempfile.mkdtemp()
    try:
        with open(os.path.join(d, "torch.py"), "w", encoding="utf-8") as f:
            f.write(fake_torch_src)
        env = {**os.environ, "PYTHONPATH": d}
        r = subprocess.run([sys.executable, "-c", gpus._GPU_PROBE],
                           capture_output=True, text=True, env=env, timeout=60)
        return json.loads(r.stdout)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_probe_no_torch():
    assert _run_probe(_FAKE_TORCH_IMPORT_FAILS) == []   # import failure -> "[]"


def test_probe_no_gpu():
    assert _run_probe(_FAKE_TORCH_NO_GPU) == []         # torch present, no CUDA/ROCm device


def test_probe_enumerates():
    # name enriched with gfx arch (":" suffix stripped) + VRAM, since ROCm reports
    # every AMD card under the same generic name — arch+VRAM is what disambiguates.
    assert _run_probe(_FAKE_TORCH_2GPU) == [
        {"index": 0, "name": "GPU-0 (gfx906, 32GB)"},
        {"index": 1, "name": "GPU-1 (gfx1200, 16GB)"},
    ]


# --- the wrapper: parses probe stdout, swallows failures ---------------------

def _list_gpus_with_run(returncode, stdout):
    """list_gpus() with subprocess.run stubbed to a canned result; cache cleared
    around it so a stub can't leak into a later real call."""
    orig = gpus.subprocess.run
    list_gpus.cache_clear()
    try:
        gpus.subprocess.run = lambda *a, **k: types.SimpleNamespace(returncode=returncode, stdout=stdout)
        return list_gpus()
    finally:
        gpus.subprocess.run = orig
        list_gpus.cache_clear()


def test_list_gpus_parses_probe_json():
    assert _list_gpus_with_run(0, '[{"index": 0, "name": "GPU-0 (gfx906, 32GB)"}]') == [
        {"index": 0, "name": "GPU-0 (gfx906, 32GB)"},
    ]


def test_list_gpus_swallows_probe_failure():
    assert _list_gpus_with_run(1, "") == []             # non-zero exit -> []
    assert _list_gpus_with_run(0, "") == []             # empty stdout -> []


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
    """installed_torch_build with a stubbed torch exposing version.hip/cuda. This
    one stays in-process: reading a version string does not init HIP, so it never
    pins a GPU (unlike list_gpus, which runs actual device enumeration)."""
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
    test_probe_no_torch,
    test_probe_no_gpu,
    test_probe_enumerates,
    test_list_gpus_parses_probe_json,
    test_list_gpus_swallows_probe_failure,
    test_detect_gpu_vendor,
    test_installed_torch_build,
]

if __name__ == "__main__":
    sys.exit(run(TESTS, "test_gpus"))
