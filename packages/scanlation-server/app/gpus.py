"""Host GPU inventory for the admin device picker.

Lives in the server core (NOT scanlation_sdk.device — that module must not be
imported by the core; a stale /plugins copy shadows it). ``list_gpus`` probes
torch in a THROWAWAY subprocess, never in this process: a ROCm/CUDA context
initialized here would open the GPU's kfd/render node for the server's whole
lifetime and pin the card at D0, blocking idle runtime-suspend (D3cold ~0W).
``installed_torch_build`` may import torch in-process — reading a version string
does not init HIP, so it doesn't pin anything. Both are cached / lazy so nothing
drags torch into startup or the fast test suite.
"""
import glob
import json
import os
import subprocess
import sys
from functools import lru_cache

# Enumerate GPUs in a child that exits immediately, so this long-lived process
# never holds a HIP/CUDA context (that would pin the card at D0). Prints a JSON
# [{index, name}]; "[]" on no-torch / no-GPU / any failure.
_GPU_PROBE = r"""
import json
try:
    import torch
    out = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            arch = (getattr(p, "gcnArchName", "") or "").split(":")[0]  # gfx906:sramecc+:xnack- -> gfx906
            tags = ", ".join(t for t in (arch, f"{round(p.total_memory / 1024**3)}GB") if t)
            out.append({"index": i, "name": f"{p.name} ({tags})" if tags else p.name})
    print(json.dumps(out))
except Exception:
    print("[]")
"""


@lru_cache(maxsize=1)
def list_gpus() -> list[dict]:
    """[{index, name}] for every visible CUDA/ROCm device, or [] if torch is
    absent / reports no GPU. A ROCm torch build reports its AMD GPUs here too.

    The name is enriched with gfx arch + VRAM: ROCm reports every AMD card as the
    generic "AMD Radeon Graphics", so two AMD GPUs are indistinguishable by name
    alone — arch (gfx906/gfx1200) + VRAM is what tells them apart in the picker.
    Display-only; the picker still selects by index (cuda:N).

    Probed in a throwaway subprocess (see _GPU_PROBE): initializing HIP/CUDA in
    this process would pin the card at D0 for the server's lifetime and block idle
    runtime-suspend. The child inherits our env + sys.path (so it finds the same
    torch and enumerates in the same cuda:N order the picker selects by), then
    exits and releases the GPU."""
    try:
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
        r = subprocess.run([sys.executable, "-c", _GPU_PROBE],
                           capture_output=True, text=True, timeout=60, env=env)
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else []
    except Exception:  # noqa: BLE001 - probe failure -> no GPUs
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
