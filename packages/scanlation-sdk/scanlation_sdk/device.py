"""Torch compute-device policy shared by local-model engines.

Plugin-facing only: the server core must not import this module (a stale sdk
copy on the /plugins volume shadows the core's — see plugins_install docs).
torch is imported lazily so the SDK keeps its thin dependency set.
"""
from __future__ import annotations


def pick_device(hint: str) -> str:
    """Resolve a device hint to an actual torch device: 'cpu' pins CPU; 'cuda' or
    'cuda:N' means GPU — cuda (optionally the Nth device) if actually available,
    else CPU fallback. An out-of-range/unparseable index falls back to the default
    GPU. The hint is an engine's per-engine override or its DEFAULT_DEVICE (there
    is no global device). (A ROCm torch build reports cuda.)"""
    hint = hint.lower()
    if hint == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            if hint.startswith("cuda:"):
                try:
                    idx = int(hint.split(":", 1)[1])
                except ValueError:
                    return "cuda"
                if 0 <= idx < torch.cuda.device_count():
                    return f"cuda:{idx}"
            return "cuda"  # bare "cuda" or out-of-range index -> default GPU
    except Exception:  # noqa: BLE001 - no torch -> CPU
        pass
    return "cpu"


def device_label(device: str) -> str:
    """User-facing label for a resolved torch device: 'cpu' -> 'CPU', 'cuda' ->
    'GPU', 'cuda:N' -> 'GPU N' (matches the admin UI's CPU/GPU wording). Anything
    else -> uppercased."""
    device = device.lower()
    if device.startswith("cuda:"):
        return f"GPU {device.split(':', 1)[1]}"
    return {"cpu": "CPU", "cuda": "GPU"}.get(device, device.upper())


def release_cuda_cache() -> None:
    """Free cached VRAM after an unload; silent no-op without torch/GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
