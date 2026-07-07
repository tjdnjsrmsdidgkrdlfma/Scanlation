"""Host GPU inventory for the admin device picker.

Lives in the server core (NOT scanlation_sdk.device — that module must not be
imported by the core; a stale /plugins copy shadows it) and imports torch
directly. Enumeration is lazy (import inside the function) so importing this
module never drags torch into startup or the fast test suite, and cached because
GPUs don't hot-plug and /get_settings/ is polled.
"""
import glob
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def list_gpus() -> list[dict]:
    """[{index, name}] for every visible CUDA/ROCm device, or [] if torch is
    absent / reports no GPU. A ROCm torch build reports its AMD GPUs here too."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        return [
            {"index": i, "name": torch.cuda.get_device_name(i)}
            for i in range(torch.cuda.device_count())
        ]
    except Exception:  # noqa: BLE001 - no torch / probe failure -> no GPUs
        return []


def detect_gpu_vendor() -> str | None:
    """Which GPU vendor is passed through to this container, from device nodes (no
    torch needed): /dev/kfd -> "amd" (ROCm), /dev/nvidia* -> "nvidia" (CUDA), both
    present -> "both", neither -> None. Used to auto-pick the torch wheel index at
    plugin-install time (torch is one build = one vendor)."""
    amd = os.path.exists("/dev/kfd")
    nvidia = bool(glob.glob("/dev/nvidia[0-9]*"))
    if amd and nvidia:
        return "both"
    if amd:
        return "amd"
    if nvidia:
        return "nvidia"
    return None


def installed_torch_build() -> str | None:
    """The build of the torch actually installed in /plugins: "rocm" / "cuda" /
    "cpu", or None if torch isn't installed. Lets /admin warn when the selected
    backend doesn't match what's installed. torch imported lazily; None on
    absence/failure."""
    try:
        import torch
        if getattr(torch.version, "hip", None):
            return "rocm"
        if getattr(torch.version, "cuda", None):
            return "cuda"
        return "cpu"
    except Exception:  # noqa: BLE001 - no torch -> not installed
        return None
