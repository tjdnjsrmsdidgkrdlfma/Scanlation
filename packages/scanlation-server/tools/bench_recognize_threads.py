#!/usr/bin/env python3
"""Benchmark: manga-ocr CPU recognize -- all-cores-serial vs many-single-thread.

Just run it. No arguments needed:

    python tools/bench_recognize_threads.py

It makes its own test crops, sweeps a few thread layouts, prints a table, and
writes a report file (bench_report_<time>.md) next to where you ran it.

It answers two things:
  1. Give ONE inference all cores and run them serially (what the pipeline does
     now), or run MANY single-thread workers in parallel?
  2. Size the pool to PHYSICAL cores or to LOGICAL (hyperthread) count?

Needs the manga-ocr weights (~400MB) installed, so run it where the models live.
Everything else is optional:
    --data DIR    use real images instead of synthetic ($BENCH_DATA also sets this)
    --detect      with --data pages, cut real bubble crops via the detector
    --items N     recognize calls per layout (default 120)
    --workers ..  custom sweep, e.g. "1x0,2x1,8x1,16x1"  (Nx0 = N workers/all cores)

torch/manga_ocr are imported only inside worker processes (after the thread count
is pinned), so each worker's thread setting actually takes effect and Windows
spawn stays happy.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

# --- per-process worker state (one set per pool process) ---
_MODEL = None
_CACHE: dict = {}


@contextlib.contextmanager
def _silenced():
    """Send this process's stdout+stderr to devnull at the fd level for the block,
    then restore. Model loaders (loguru 'Using CPU', tqdm 'Loading weights', the HF
    hub warning) write to those fds; workers/childs return data over pipes, not
    stdout, so nothing useful is lost -- only the chatter. Restored on exit so a
    genuine error afterwards is still visible."""
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


def _init(threads: int, force_cpu: bool, idx_q) -> None:
    """Pool-process setup. Pin the thread count BEFORE importing torch, optionally
    pin CPU affinity to a distinct logical CPU, then load the model once so the
    timed region is warm."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence + avoid tokenizer fork deadlock

    try:  # give each worker its own logical CPU so PHYS/LOGICAL land where we think
        import psutil  # type: ignore
        if idx_q is not None:
            idx = idx_q.get_nowait()
            ncpu = psutil.cpu_count(logical=True) or 1
            psutil.Process().cpu_affinity([idx % ncpu])
    except Exception:  # noqa: BLE001 - pinning is a nicety, never fatal
        pass

    with _silenced():  # the model loaders are the noisy part
        import torch
        torch.set_num_threads(threads)
        from manga_ocr import MangaOcr  # lazy: torch + transformers
        global _MODEL
        _MODEL = MangaOcr(force_cpu=force_cpu)


def _recognize(path: str) -> int:
    """One recognize call. The decoded image is cached per process so we time the
    model, not disk I/O (the crop set is small and replicated to reach --items)."""
    from PIL import Image
    img = _CACHE.get(path)
    if img is None:
        img = Image.open(path).convert("RGB")
        _CACHE[path] = img
    _MODEL(img)  # type: ignore[misc]
    return 1


def _run_config(workers: int, threads: int, force_cpu: bool,
                work: list, warmup: list, pin: bool) -> tuple[float, float]:
    """Run one (workers x threads) layout; return (crops/sec, seconds) over `work`
    measured warm (after every worker has spun up + loaded the model)."""
    import multiprocessing as mp
    idx_q, mgr = None, None
    if pin:
        try:
            mgr = mp.Manager()
            idx_q = mgr.Queue()
            for i in range(workers):
                idx_q.put(i)
        except Exception:  # noqa: BLE001
            idx_q = None

    ex = ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(threads, force_cpu, idx_q))
    try:
        # chunksize=1 over >=workers tasks makes every worker spin up + warm before
        # we start the clock, so model-load time never lands inside the timing.
        list(ex.map(_recognize, warmup, chunksize=1))
        chunk = max(1, len(work) // (workers * 4))
        t0 = time.perf_counter()
        list(ex.map(_recognize, work, chunksize=chunk))
        dt = time.perf_counter() - t0
    finally:
        ex.shutdown(wait=True)
        if mgr is not None:
            mgr.shutdown()
    return len(work) / dt, dt


# --- crop sources ------------------------------------------------------------
_JP = ("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよ"
       "ロボットマンガアニメドカーンバキュウゴゴゴ"
       "日本語漫画声人大小上中下口目手火水木")


def _synthetic_crops(n: int) -> list[str]:
    """Render n little Japanese text crops to a temp dir (zero-config default).
    Fixed seed -> the same crops every run, so numbers are comparable over time."""
    import random
    from PIL import Image, ImageDraw, ImageFont
    rng = random.Random(0)

    font_path = None
    for cand in ("C:/Windows/Fonts/msgothic.ttc", "C:/Windows/Fonts/YuGothM.ttc",
                 "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/malgun.ttf",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if os.path.exists(cand):
            font_path = cand
            break

    tmp = Path(tempfile.mkdtemp(prefix="bench_synth_"))
    fs = 30
    font = ImageFont.truetype(font_path, fs) if font_path else ImageFont.load_default()
    paths = []
    for i in range(n):
        s = "".join(rng.choice(_JP) for _ in range(rng.randint(3, 11)))
        im = Image.new("RGB", (fs * len(s) + 20, fs + 20), "white")
        ImageDraw.Draw(im).text((10, 10), s, fill="black", font=font)
        p = tmp / f"synth_{i:03d}.png"
        im.save(p)
        paths.append(str(p))
    return paths


def _detected_crops(files: list[Path]) -> list[str]:
    """Run the detector once and save each bbox crop to a temp dir (realistic)."""
    from PIL import Image
    from scanlation_comic_text_and_bubble_detector.plugin import ComicTextAndBubbleDetector
    tmp = Path(tempfile.mkdtemp(prefix="bench_detect_"))
    det = ComicTextAndBubbleDetector()
    det.load()
    crops: list[str] = []
    try:
        for page in files:
            img = Image.open(page).convert("RGB")
            for i, r in enumerate(det.detect(img, {})):
                x0, y0, x1, y1 = (int(v) for v in r.bbox)
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                out = tmp / f"{page.stem}_{i}.png"
                img.crop((x0, y0, x1, y1)).save(out)
                crops.append(str(out))
    finally:
        det.unload()
    if not crops:
        sys.exit("detector produced no crops")
    return crops


def _detect_child(files, q) -> None:
    """Run detection and hand the crop paths back over `q`. Runs in a throwaway
    subprocess (see _detect_isolated) so torch stays out of the parent."""
    try:
        with _silenced():  # detector load + transformers chatter
            crops = _detected_crops(files)
        q.put(("ok", crops))
    except BaseException as e:  # noqa: BLE001 - ship any failure across the boundary
        q.put(("err", f"{type(e).__name__}: {e}"))


def _detect_isolated(files) -> list:
    """Detect crops in a forked subprocess so the parent -- which later forks the
    recognize pool -- never imports torch. Forking a process that has already
    initialised torch deadlocks the child on inherited thread-pool locks; that is
    exactly what hung the in-process --detect path."""
    import multiprocessing as mp
    try:
        ctx = mp.get_context("fork")
    except ValueError:  # no fork (Windows); --detect is a Linux-server path anyway
        ctx = mp.get_context()
    q = ctx.Queue()
    p = ctx.Process(target=_detect_child, args=(files, q))
    p.start()
    status, payload = q.get()
    p.join()
    if status == "err":
        sys.exit(f"detect failed: {payload}")
    return payload


def _build_crops(data, use_detect: bool) -> tuple[list[str], str]:
    """Return (crop paths, human source label). No --data -> synthetic."""
    if data is None:
        return _synthetic_crops(24), "synthetic"
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted(p for p in Path(data).rglob("*") if p.suffix.lower() in exts)
    if not files:
        sys.exit(f"no images found under {data}")
    if use_detect:
        return _detect_isolated(files), f"detected from {len(files)} pages"
    return [str(p) for p in files], f"{len(files)} image files"


def _cycle_to(items: list[str], n: int) -> list[str]:
    return [items[i % len(items)] for i in range(n)] if items else []


def _parse_sweep(spec: str, logical: int) -> list[tuple[int, int]]:
    out = []
    for tok in spec.split(","):
        w, t = tok.lower().split("x")
        out.append((int(w), logical if int(t) == 0 else int(t)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.getenv("BENCH_DATA"),
                    help="folder of crops (or pages with --detect); default: $BENCH_DATA, else synthetic")
    ap.add_argument("--detect", action="store_true",
                    help="with --data pages, cut real bubble crops via the detector")
    ap.add_argument("--items", type=int, default=120, help="recognize calls per layout")
    ap.add_argument("--workers", default="", help='custom sweep, e.g. "1x0,2x1,8x1,16x1"')
    args = ap.parse_args()

    logical = os.cpu_count() or 1
    physical, has_psutil = None, False
    try:
        import psutil
        has_psutil = True
        physical = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001
        pass
    phys_note = ""
    if not physical:
        physical = max(1, logical // 2)  # guess: assume SMT2
        phys_note = " (guessed; pip install psutil for exact + pinning)"

    if args.workers:
        sweep = _parse_sweep(args.workers, logical)
    else:
        # baseline (1 worker, all cores) + a 1-thread-per-worker scaling curve
        # (powers of two up to the logical count, plus the physical & logical marks)
        pts, c = {physical, logical}, 1
        while c <= logical:
            pts.add(c)
            c *= 2
        sweep = [(1, logical)] + [(w, 1) for w in sorted(pts) if w <= logical]
        # + 2-thread-per-worker at the physical marks, so "same cores, 1t vs 2t"
        # is visible: 8wx2t vs 16wx1t (both 16 cores), 4wx2t vs 8wx1t (both 8).
        sweep += [(w, 2) for w in sorted({max(1, physical // 2), physical})]

    crops, source = _build_crops(args.data, args.detect)
    work = _cycle_to(crops, args.items)
    pin = has_psutil

    rows = [
        "# manga-ocr CPU recognize -- thread-layout benchmark",
        "",
        f"- when: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- cpu: {logical} logical / {physical} physical{phys_note}",
        f"- work: {len(work)} recognize calls over {len(crops)} crops ({source}), force_cpu, pin={pin}",
        "",
        "| layout | cores | crops/sec | sec | vs base |",
        "|---|---|---|---|---|",
    ]
    print("\n".join(rows[:5]))
    print(f"\n{'layout':>14} {'cores':>6} {'crops/sec':>10} {'sec':>7} {'vs base':>8}")
    print("-" * 50)

    base = None
    for workers, threads in sweep:
        rate, dt = _run_config(workers, threads, True, work,  # force_cpu: this bench is CPU-only
                               _cycle_to(crops, workers * 2), pin)
        base = base or rate
        layout = f"{workers}w x {threads}t"
        print(f"{layout:>14} {workers * threads:>6} {rate:>10.2f} {dt:>7.2f} {rate / base:>7.2f}x")
        rows.append(f"| {layout} | {workers * threads} | {rate:.2f} | {dt:.2f} | {rate / base:.2f}x |")

    rows += [
        "",
        "## how to read",
        "- `1w x <all>t` is today's pipeline behaviour (one inference on all cores, serial).",
        "- PHYSx1 vs LOGICALx1 is the cores-vs-hyperthread answer:",
        "  - LOGICALx1 ~= PHYSx1  -> compute/SIMD-bound; physical cores are the unit.",
        "  - LOGICALx1 >  PHYSx1  -> memory-stall hiding pays; hyperthreads help.",
        "- the curve flattens once memory bandwidth saturates -- that knee is the real",
        "  ceiling, not the core count.",
    ]

    name = f"bench_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    body = "\n".join(rows) + "\n"
    # cwd first (nice on a normal checkout); fall back to the temp dir so a
    # read-only cwd (e.g. `/` inside the container as a non-root user) can't throw
    # away a finished run.
    for target in (Path.cwd() / name, Path(tempfile.gettempdir()) / name):
        try:
            target.write_text(body, encoding="utf-8")
            print(f"\nreport written: {target}")
            return 0
        except OSError as e:  # noqa: PERF203
            print(f"(could not write {target}: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
