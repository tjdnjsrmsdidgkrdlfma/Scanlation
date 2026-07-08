"""Shared helpers for the ``tools/bench_recognize_*.py`` benchmarks.

Only what is genuinely identical across the benches lives here: the fd-level
silencer, the deskewing crop loader, GPU device selection, the PaddleOCR-VL
loader, and the report writer.

Two invariants this module must not break:

* **Nothing heavy at import time.** PIL, torch, ``app`` and the engine plugins are
  imported inside the functions that need them. ``bench_recognize_threads.py``
  pins its thread count before torch is ever imported, and its pool workers
  re-import their module -- a torch at module scope would defeat both.
* **No ``_bootstrap``.** Importing this module must not rewrite ``sys.path`` or
  reconfigure stdio. ``deskewed_crops`` does reach into ``app.geometry``, so a
  caller that uses it imports ``_bootstrap`` itself (batch/gpuconc do).
"""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# The image types the benches accept as pages or pre-cut crops.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@contextlib.contextmanager
def silenced():
    """Send this process's stdout+stderr to devnull at the fd level for the block,
    then restore. Model loaders (loguru, tqdm 'Loading weights', HF hub warnings)
    write to those fds; pool workers hand results back over pipes, not stdout, so
    nothing useful is lost -- only the chatter. Restored on exit so a genuine error
    afterwards is still visible."""
    save1, save2 = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(save1, 1)
        os.dup2(save2, 2)
        for fd in (devnull, save1, save2):
            os.close(fd)


# --- crop sources ------------------------------------------------------------
def deskewed_crops(files):
    """Detect + deskew each region on the pages -> the exact upright crops the
    recognizer sees in production (app.pipeline.detect_and_recognize does the same
    detect -> deskew_crop), so the straggler/probe numbers are real.

    Runs the detector in-process. A bench that later forks a worker pool must NOT
    use this -- forking a process that has initialised torch deadlocks the child
    (see bench_recognize_threads._raw_bbox_crop_files, which cuts plain bbox crops
    in an isolated subprocess instead)."""
    from PIL import Image
    from app.geometry import deskew_crop
    from scanlation_comic_text_and_bubble_detector.plugin import ComicTextAndBubbleDetector
    det = ComicTextAndBubbleDetector()
    with silenced():
        det.load()
    crops = []
    try:
        for page in files:
            img = Image.open(page).convert("RGB")
            for r in det.detect(img, {}):
                x0, y0, x1, y1 = (int(v) for v in r.bbox)
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                crops.append(deskew_crop(img, r))
    finally:
        det.unload()
    if not crops:
        sys.exit("detector produced no crops")
    return crops


def load_crops(data, use_detect: bool):
    """Return (list[PIL.Image], human source label) from a pages/crops folder, or a
    single image file (handy for a quick one-page probe when the data dir is
    read-only and a throwaway one-image folder isn't an option)."""
    from PIL import Image
    root = Path(data)
    files = [root] if root.is_file() else sorted(
        f for f in root.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit(f"no images found under {data}")
    if use_detect:
        return deskewed_crops(files), f"detected from {len(files)} pages"
    return [Image.open(f).convert("RGB") for f in files], f"{len(files)} image files"


# --- engines -----------------------------------------------------------------
def paddle_device(force_cpu: bool) -> tuple[str | None, str]:
    """Pick the device for the PaddleOCR-VL half and, on a skip, say WHY in a way
    that's actionable on this project's boxes. Returns (device, reason): device is
    None -> skip (reason explains it); else the torch device to run on.

    The old message just said 'no CUDA GPU', a dead end on an AMD host -- it didn't
    distinguish a CPU-only torch (reinstall with 백엔드=GPU so the ROCm wheel is
    pulled) from a GPU-capable build that still sees no device (passthrough / gfx
    override). A ROCm torch reports as cuda (see sdk device.py)."""
    if force_cpu:
        return "cpu", ""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - no torch -> no engine installed yet
        return None, f"torch not importable ({exc}); install an engine first"
    if torch.cuda.is_available():
        return "cuda", ""  # a ROCm build reports True here too -> AMD GPU
    hip, cuda = torch.version.hip, torch.version.cuda
    if hip:
        reason = (f"torch is a ROCm build (hip {hip}) but no GPU is visible -- check device "
                  "passthrough (docker-compose.rocm.yml: /dev/kfd, /dev/dri) and, for "
                  "RDNA4/gfx1200, set HSA_OVERRIDE_GFX_VERSION")
    elif cuda:
        reason = (f"torch is a CUDA build (cuda {cuda}) but no GPU is visible -- check the "
                  "NVIDIA container runtime / device passthrough")
    else:
        reason = ("torch is a CPU-only build -- reinstall PaddleOCR-VL with 백엔드=GPU in "
                  "/admin so the ROCm/CUDA torch wheel is pulled")
    return None, reason


def load_paddle(device: str, attn: str | None):
    """Load the PaddleOCR-VL recognizer (one model copy), optionally forcing an
    attention backend (BENCH_ATTN). Shared by the batch probe, the concurrency pool
    workers, and the in-process profiles.

    ``_device_override`` is a declared SDK field (LocalModelEngineBase); ``_repo()``
    and ``_model`` are the plugin's internals -- reaching for them is what makes this
    a bench and not a user. Keeping that reach in one place is the point."""
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()  # silence per-call max_length/max_new_tokens warnings
    from scanlation_paddleocr_vl_for_manga.plugin import PaddleOcrVLForMangaRecognizer

    rec = PaddleOcrVLForMangaRecognizer()
    rec._device_override = device
    with silenced():
        rec.load()
        if attn:
            # Reload the just-loaded model with the override so the installed plugin
            # needn't change. A remote-code model without support raises ValueError
            # here -> that's the finding.
            from transformers import AutoModelForImageTextToText
            rec._model = AutoModelForImageTextToText.from_pretrained(
                rec._repo(), torch_dtype="auto", device_map=device,
                local_files_only=True, attn_implementation=attn).eval()
    return rec


# --- report ------------------------------------------------------------------
def write_report(rows: list[str], prefix: str, *, dump: bool = True) -> int:
    """Write the markdown report and return a process exit code.

    With ``dump``, the whole report also goes to stdout first: under Docker the cwd
    and /tmp are often unwritable, so stdout is the only guaranteed sink and the file
    is a best-effort convenience on top, never the sole copy. The cwd is tried first
    (nice on a normal checkout), then the temp dir, so a read-only cwd can't throw
    away a finished run."""
    name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    body = "\n".join(rows) + "\n"
    if dump:
        print("\n" + "=" * 72 + "\nFULL REPORT (copy from here if the file below didn't write)\n" + "=" * 72)
        print(body)
    for target in (Path.cwd() / name, Path(tempfile.gettempdir()) / name):
        try:
            target.write_text(body, encoding="utf-8")
            print(f"report written: {target}")
            return 0
        except OSError as e:  # noqa: PERF203
            print(f"(could not write {target}: {e})")
    if dump:
        print("(report file not written -- use the FULL REPORT block above)")
    return 0
