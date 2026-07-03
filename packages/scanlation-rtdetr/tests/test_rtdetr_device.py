"""RT-DETR device selection — model-free (no weights; torch optional).

Proves _pick_device honors context.device (the /admin-set, SCANLATION_DEVICE-
seeded hint) instead of always grabbing cuda, so one switch moves detector +
recognizer together.
"""
from __future__ import annotations

from scanlation_sdk.context import context
from scanlation_rtdetr.plugin import RTDetrDetector


def test_pick_device_cpu_is_pinned():
    """device 'cpu' -> always CPU, regardless of GPU presence (case-insensitive)."""
    saved = context.device
    try:
        context.device = "cpu"
        assert RTDetrDetector._pick_device() == "cpu"
        context.device = "CPU"
        assert RTDetrDetector._pick_device() == "cpu"
    finally:
        context.device = saved


def test_pick_device_gpu_uses_cuda_when_available():
    """device != 'cpu' -> cuda if torch reports it, else a safe CPU fallback.
    Computed against the same torch check so it's deterministic on any host."""
    saved = context.device
    try:
        context.device = "cuda"
        expected = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                expected = "cuda"
        except Exception:  # noqa: BLE001 - no torch -> stays cpu
            pass
        assert RTDetrDetector._pick_device() == expected
    finally:
        context.device = saved


TESTS = [
    test_pick_device_cpu_is_pinned,
    test_pick_device_gpu_uses_cuda_when_available,
]

if __name__ == "__main__":
    import sys

    from scanlation_sdk.testing import run

    sys.exit(run(TESTS, "test_rtdetr_device"))
