#!/usr/bin/env python3
"""Benchmark: PaddleOCR-VL GPU multi-worker concurrency -- does running several
B=1 recognizes at once fill the idle GPU that a single request leaves, and by
how much, without the ragged-padding correctness break that killed crop batching?

Batching (bench_recognize_batch.py) stacks a page's crops into one forward: it
found ~2x of idle-GPU headroom at B=2 but regressed at B>=8 and, worse, went
silently wrong (dynamic-res ragged vision tokens -> left-pad -> position_ids
corruption). Concurrency is the other way to spend that same idle GPU: N workers
each doing a clean B=1 forward. No padding -> no correctness break; the only
question is how much the one GPU actually overlaps concurrent work, and where
VRAM (one model copy per worker) caps the worker count.

    python tools/bench_recognize_gpu_concurrency.py PAGES_DIR --detect

(or a folder / single image of pre-cut crops without --detect; $BENCH_DATA sets
the path.) It sweeps worker counts (default 1,2,4), keeps the total timed crop
count fixed, and reports aggregate crops/sec vs the 1-worker baseline plus the
per-process VRAM peak (so W x peak estimates the total pressure). A worker count
that OOMs is caught and reported -- that ceiling is itself the finding.

Workers are separate processes (spawn), not threads: the autoregressive decode
loop holds the GIL between kernel launches, so threads would serialise on Python
and understate the gain. Processes give the GIL-free, deployment-realistic answer
(the pipeline scales manga-ocr with process workers too), at the cost of one
model copy each -- which is exactly the VRAM limit we want to measure.

Set BENCH_ATTN=sdpa to force an attention backend into each worker's load (same
knob as the batch bench). Needs the engine's weights installed + a GPU, so run it
where the models live.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 - makes `scanlation_*`/`app` importable + UTF-8 stdio

import argparse
import multiprocessing as mp
import os
import statistics
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

# Reuse the batch bench's crop cutting + GPU detection + fd silencer (importing it
# does NOT run its main -- that's guarded by __name__ == "__main__").
from bench_recognize_batch import _load_crops, _paddle_device, _silenced

# --- per-worker process globals (set once by the pool initializer) -----------
_REC = None
_CROPS = None
_REGION = None
_OPTS = None


def _load_rec(device: str, attn: str | None):
    """Load the PaddleOCR-VL recognizer (one model copy), optionally forcing an
    attention backend. Shared by the pool workers and the in-process decode profile."""
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    from scanlation_paddleocr_vl_for_manga.plugin import PaddleOcrVLForMangaRecognizer

    rec = PaddleOcrVLForMangaRecognizer()
    rec._device_override = device
    with _silenced():
        rec.load()
        if attn:  # match the batch bench's BENCH_ATTN probe
            from transformers import AutoModelForImageTextToText
            rec._model = AutoModelForImageTextToText.from_pretrained(
                rec._repo(), torch_dtype="auto", device_map=device,
                local_files_only=True, attn_implementation=attn).eval()
    return rec


def _worker_init(crops, device: str, attn: str | None, probe_cap: int) -> None:
    """Runs once in each freshly spawned worker: load the model (one copy per
    process -> the VRAM cost), then warm the kernel JIT on the first crop so the
    timed run is steady-state."""
    global _REC, _CROPS, _REGION, _OPTS
    from scanlation_sdk.contracts import Region

    _REC = _load_rec(device, attn)
    _CROPS = crops
    _REGION = Region.from_bbox(0, 0, crops[0].width, crops[0].height)  # unused by recognize
    _OPTS = {"max_new_tokens": probe_cap}
    with _silenced():
        _REC.recognize(_CROPS[0], _REGION, _OPTS)  # warmup (JIT/kernel cache)


def _worker_task(i: int) -> tuple[float, int, str]:
    """One B=1 recognize; returns (wall ms, peak torch VRAM, recognized text).
    The text is the tie-breaker: a slow crop whose output is one token/phrase
    repeated to the cap is runaway generation; varied long text is genuine. Peak
    VRAM is torch-allocator only -- add HIP context + MIOpen, so a lower bound."""
    import torch
    t0 = time.perf_counter()
    text = _REC.recognize(_CROPS[i % len(_CROPS)], _REGION, _OPTS)
    ms = (time.perf_counter() - t0) * 1000
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return ms, peak, text


def bench_concurrency(crops, worker_counts, device, items, attn, probe_cap, rows) -> None:
    ctx = mp.get_context("spawn")  # fork + CUDA/HIP is unsafe
    base_rate = None

    print(f"\n-- PaddleOCR-VL {device}: multi-worker concurrency (each worker B=1)")
    print(f"{'workers':>7} {'ran':>5} {'crops/sec':>10} {'speedup':>8} {'VRAM/proc':>10}")
    rows += ["### PaddleOCR-VL -- GPU multi-worker concurrency", "",
             "Each worker is a separate process running B=1 recognizes (no batching, "
             "no padding). crops/sec = aggregate over `items` timed recognizes split "
             "across the workers; speedup vs 1 worker. est VRAM = per-process torch "
             "peak x W (lower bound; add HIP context + MIOpen).", "",
             "| workers | crops/sec | speedup | est VRAM (peak x W) |",
             "|---|---|---|---|"]

    for w in worker_counts:
        try:
            with ProcessPoolExecutor(max_workers=w, mp_context=ctx, initializer=_worker_init,
                                     initargs=(crops, device, attn, probe_cap)) as ex:
                # Prime: force all W workers to spin up + load + warm before timing.
                # (2*W tasks reliably reaches every worker; they persist afterwards.)
                list(ex.map(_worker_task, range(max(2 * w, 4))))
                t0 = time.perf_counter()
                out = list(ex.map(_worker_task, range(items)))
                dt = time.perf_counter() - t0
            percrop = [r[0] for r in out]        # per-recognize wall ms (in-worker)
            peaks = [r[1] for r in out]
            texts = [r[2] for r in out]
            rate = items / dt
            if base_rate is None:
                base_rate = rate
            speed = rate / base_rate
            vram = (max(peaks) / 1e9) if peaks else 0.0
            print(f"{w:>7} {'yes':>5} {rate:>10.2f} {speed:>7.2f}x {vram:>7.2f}GB")
            print(f"        per-crop ms: min {min(percrop):.0f} / med {statistics.median(percrop):.0f}"
                  f" / max {max(percrop):.0f}")
            # slowest first, each with its output text: repeated token/phrase to the
            # cap = runaway; varied text = genuinely long. chars = len(decoded string).
            print("        ms · chars · text  (slowest first):")
            for ms_i, txt in sorted(zip(percrop, texts), key=lambda r: -r[0]):
                snippet = txt if len(txt) <= 70 else txt[:70] + "…"
                print(f"          {ms_i:>7.0f} · {len(txt):>4} · {snippet!r}")
            rows.append(f"| {w} | {rate:.2f} | {speed:.2f}x | ~{vram * w:.1f}GB ({vram:.1f}×{w}) |")
            rows.append(f"| | per-crop ms min {min(percrop):.0f}/med {statistics.median(percrop):.0f}"
                        f"/max {max(percrop):.0f}, chars max {max(len(t) for t in texts)} | | |")
        except Exception as exc:  # noqa: BLE001 - OOM / driver limit at high W is a valid finding
            print(f"{w:>7} {'no':>5}   {type(exc).__name__}: {exc}")
            rows.append(f"| {w} | no | {type(exc).__name__}: {exc} | |")
            break  # a higher worker count won't fare better
    rows += [""]


def bench_decode_profile(crops, device: str, cap: int, attn, n_crops: int, rows) -> None:
    """Time each generation STEP on the biggest crops (+ the smallest for contrast).
    Flat per-token = every step re-does the same work (vision/prefill recomputed ->
    no effective KV reuse). Growing per-token = O(n) attention over a lengthening
    sequence. The first step also carries prefill. Runs in-process (no pool)."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    rec = _load_rec(device, attn)
    proc, model = rec._proc, rec._model

    class _StepTimer(StoppingCriteria):  # records a timestamp after each decoded token
        def __init__(self):
            self.t = []

        def __call__(self, input_ids, scores, **kw):
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # the step's GPU work is done before we stamp
            self.t.append(time.perf_counter())
            return False  # never stop early

    by_area = sorted(range(len(crops)), key=lambda i: crops[i].width * crops[i].height, reverse=True)
    picks = by_area[:n_crops] + by_area[-1:]  # biggest N + the smallest, for contrast

    print(f"\n-- decode profile (per-step ms), device {device}, cap {cap}")
    rows += ["### PaddleOCR-VL -- decode profile (per-step ms)", "",
             "Wall time of each generation step on the biggest crops (+ smallest). "
             "Flat steady/token = each step re-does the same work (vision/prefill "
             "recomputed, no KV reuse). Growing = O(n) attention. First step carries "
             "prefill.", "",
             "| crop px | tokens | prefill+t1 ms | steady/token ms | first steps ms |",
             "|---|---|---|---|---|"]
    for idx in picks:
        crop = crops[idx].convert("RGB") if crops[idx].mode != "RGB" else crops[idx]
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": rec.PROMPT}]}]
        text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=[text], images=[crop], return_tensors="pt").to(model.device)
        timer = _StepTimer()
        t0 = time.perf_counter()
        with torch.no_grad(), _silenced():
            model.generate(**inputs, max_new_tokens=cap, do_sample=False,
                           stopping_criteria=StoppingCriteriaList([timer]))
        ts = timer.t
        if not ts:
            continue
        deltas = [(ts[0] - t0) * 1000] + [(ts[i] - ts[i - 1]) * 1000 for i in range(1, len(ts))]
        steady = statistics.median(deltas[1:]) if len(deltas) > 1 else float("nan")
        px = crop.width * crop.height
        firsts = " ".join(f"{d:.0f}" for d in deltas[:8])
        print(f"  {crop.width}x{crop.height} ({px // 1000}k px), {len(deltas)} tok: "
              f"prefill+t1={deltas[0]:.0f}ms  steady/token med={steady:.0f}ms")
        print(f"    first steps ms: {firsts}")
        rows.append(f"| {crop.width}x{crop.height} ({px // 1000}k) | {len(deltas)} | "
                    f"{deltas[0]:.0f} | {steady:.0f} | {firsts} |")
    rows += [""]


def _cap_pixels(crops, max_px: int):
    """Downscale crops whose area exceeds max_px (aspect preserved) so the dynamic-res
    processor emits fewer vision tokens -- the workaround for eager O(n^2) attention
    when flash isn't available. Returns (new_crops, n_resized)."""
    from PIL import Image
    out, n = [], 0
    for c in crops:
        if max_px and c.width * c.height > max_px:
            scale = (max_px / (c.width * c.height)) ** 0.5
            out.append(c.resize((max(1, int(c.width * scale)), max(1, int(c.height * scale))), Image.LANCZOS))
            n += 1
        else:
            out.append(c)
    return out, n


def bench_pixel_sweep(crops, device: str, out_cap: int, attn, sweep, items: int, rows) -> None:
    """Sweep max-pixels caps on the same crops, comparing each capped read to the
    UNCAPPED read (char-level SequenceMatcher ratio) and timing it -- the accuracy-vs-
    speed knee for choosing a production downscale cap. Single process, warmed so the
    one-time first-forward JIT doesn't land on the first cap and skew it."""
    import difflib
    from scanlation_sdk.contracts import Region

    rec = _load_rec(device, attn)
    region = Region.from_bbox(0, 0, crops[0].width, crops[0].height)  # unused by recognize
    opts = {"max_new_tokens": out_cap}
    sample = crops[:max(1, items)]
    with _silenced():
        rec.recognize(sample[0], region, opts)  # warmup: absorb first-forward JIT

    refs = [rec.recognize(c, region, opts) for c in sample]  # uncapped reference reads

    print(f"\n-- max-pixels sweep (accuracy vs speed), device {device}, out-cap {out_cap}, {len(sample)} crops")
    print(f"{'pixels':>10} {'crops/sec':>10} {'exact':>9} {'char-sim':>9}")
    rows += ["### PaddleOCR-VL -- max-pixels sweep (accuracy vs speed)", "",
             "Each cap downscales crops above it, reads, and compares to the UNCAPPED "
             "read (char-level SequenceMatcher ratio). The reference is itself OCR, so "
             "this is change-from-full-res, not ground truth. Knee = highest crops/sec "
             "whose char-sim is still acceptable.", "",
             "| pixels | crops/sec | exact match | char-sim |", "|---|---|---|---|"]
    for px in sweep:
        capped = _cap_pixels(sample, px)[0] if px else sample
        t0 = time.perf_counter()
        got = [rec.recognize(c, region, opts) for c in capped]
        dt = time.perf_counter() - t0
        rate = len(sample) / dt
        exact = sum(g == r for g, r in zip(got, refs))
        sim = sum(difflib.SequenceMatcher(None, r, g).ratio() for r, g in zip(refs, got)) / len(refs)
        label = "uncapped" if not px else str(px)
        print(f"{label:>10} {rate:>10.2f} {exact:>6}/{len(refs)} {sim:>9.3f}")
        rows.append(f"| {label} | {rate:.2f} | {exact}/{len(refs)} | {sim:.3f} |")
    rows += [""]


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # live progress under a Docker pipe

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default=os.getenv("BENCH_DATA"),
                    help="folder of pages OR a single image (with --detect), or pre-cut crops; or $BENCH_DATA")
    ap.add_argument("--detect", action="store_true",
                    help="treat the folder as pages: detect + deskew real bubble crops")
    ap.add_argument("--workers", default="1,2,4", help='worker-count sweep, e.g. "1,2,4,6"')
    ap.add_argument("--items", type=int, default=24,
                    help="total timed recognizes per sweep point (split across workers)")
    ap.add_argument("--probe-cap", type=int, default=256, help="max_new_tokens per recognize")
    ap.add_argument("--paddle-cpu", action="store_true",
                    help="run on CPU anyway (smoke test only; ~60s/crop)")
    ap.add_argument("--profile-decode", action="store_true",
                    help="skip the worker sweep; profile per-step decode time on the biggest crops")
    ap.add_argument("--profile-n", type=int, default=3, help="how many of the biggest crops to profile")
    ap.add_argument("--max-pixels", type=int, default=0,
                    help="downscale crops above this pixel area before recognize (0 = off)")
    ap.add_argument("--sweep-pixels", default="",
                    help='accuracy-vs-speed: sweep max-pixels caps vs the uncapped read, e.g. "0,250000,150000,100000"')
    args = ap.parse_args()

    if not args.data:
        sys.exit("no data path: pass a folder/image or set $BENCH_DATA")

    device, reason = _paddle_device(args.paddle_cpu)
    if device is None:
        sys.exit(f"PaddleOCR-VL concurrency needs a GPU: {reason}")

    crops, src = _load_crops(args.data, args.detect)
    if args.max_pixels and not args.sweep_pixels:  # sweep controls its own capping
        crops, n_capped = _cap_pixels(crops, args.max_pixels)
        src += f", {n_capped} downscaled to <={args.max_pixels}px"
    worker_counts = [int(x) for x in args.workers.split(",") if x.strip()]
    attn = os.getenv("BENCH_ATTN")

    rows = ["# recognize GPU concurrency benchmark", "",
            f"- when: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- crops: {len(crops)} ({src})",
            f"- device: {device}" + (f", attn={attn}" if attn else ""),
            f"- worker sweep: {worker_counts}, items/point: {args.items}", ""]

    if args.sweep_pixels:
        sweep = [int(x) for x in args.sweep_pixels.split(",") if x.strip()]
        bench_pixel_sweep(crops, device, args.probe_cap, attn, sweep, args.items, rows)
    elif args.profile_decode:
        bench_decode_profile(crops, device, args.probe_cap, attn, args.profile_n, rows)
    else:
        bench_concurrency(crops, worker_counts, device, args.items, attn, args.probe_cap, rows)

    # stdout is the only guaranteed sink under Docker (cwd/tmp often unwritable).
    name = f"bench_report_gpuconc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    body = "\n".join(rows) + "\n"
    print("\n" + "=" * 72 + "\nFULL REPORT (copy from here if the file below didn't write)\n" + "=" * 72)
    print(body)
    for target in (Path.cwd() / name, Path(tempfile.gettempdir()) / name):
        try:
            target.write_text(body, encoding="utf-8")
            print(f"report written: {target}")
            return 0
        except OSError as e:  # noqa: PERF203
            print(f"(could not write {target}: {e})")
    print("(report file not written -- use the FULL REPORT block above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
