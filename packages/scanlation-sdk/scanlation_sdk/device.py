"""Torch compute-device policy shared by local-model engines.

Plugin-facing only: the server core must not import this module (a stale sdk
copy on the /plugins volume shadows the core's — see plugins_install docs).
torch is imported lazily so the SDK keeps its thin dependency set.
"""
from __future__ import annotations


def pick_device(hint: str) -> str:
    """Resolve a device hint to an actual torch device: 'cpu' pins CPU; anything
    else means GPU — cuda if actually available, else CPU fallback. The hint is
    an engine's per-engine override or its DEFAULT_DEVICE (there is no global
    device). (A ROCm torch build reports cuda.)"""
    if hint.lower() == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 - no torch -> CPU
        pass
    return "cpu"


def release_cuda_cache() -> None:
    """Free cached VRAM after an unload; silent no-op without torch/GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
