"""Host GPU inventory for the admin device picker.

Lives in the server core (NOT scanlation_sdk.device — that module must not be
imported by the core; a stale /plugins copy shadows it) and imports torch
directly. Enumeration is lazy (import inside the function) so importing this
module never drags torch into startup or the fast test suite, and cached because
GPUs don't hot-plug and /get_settings/ is polled.
"""
from __future__ import annotations

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
