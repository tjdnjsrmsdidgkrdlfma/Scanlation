#!/usr/bin/env python3
"""Probe: does torch.compile + a static KV cache cut PaddleOCR-VL's decode overhead?

recognize-decode-bound.md found the per-crop cost is ~95% autoregressive decode at
a FLAT ~64ms/token, of which weight-read is ~9% and compute <1% -- the rest is B=1
kernel-launch/dispatch overhead. torch.compile(mode="reduce-overhead") captures the
decode step into a HIP graph (no per-launch overhead) and a static KV cache gives the
decode a fixed shape so that graph is reusable. This measures whether that actually
lands on ROCm (inductor/Triton maturity + the model's mrope are the risks) by timing
the same warm crops with and without it, and CHECKS the compiled output is identical.

    python tools/bench_recognize_compile.py PAGES_DIR --detect   (or $BENCH_DATA)

Run in the container (torch lives there). The plugin applies its own 150k/pow2 cap
inside recognize(), so the shapes are production shapes.

Two honest caveats:
* **Best case only.** The warm pass runs every crop once, so the timed pass hits
  already-compiled shapes. Production feeds a FRESH shape per crop (dynamic
  resolution), so a win here only transfers if the per-shape recompile (the printed
  warm cost) is acceptable.
* **May just break.** mrope graph breaks or an immature ROCm inductor can fail the
  compile/run outright -- that negative result IS the finding, so it's caught and
  printed rather than crashing silently.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 - makes `app`/`scanlation_*` importable + UTF-8 stdio

import argparse
import os
import statistics
import sys
import time

from _bench_common import load_crops, load_paddle, paddle_device, silenced


def _pass(rec, crops, opts, label, warm_progress=False):
    """Warm every crop once (silenced -> absorbs JIT + per-shape compiles), then time
    recognize() on each. Returns (crops/sec, per-crop ms, texts, warm_seconds)."""
    from scanlation_sdk.contracts import Region

    def _region(c):
        return Region.from_bbox(0, 0, c.width, c.height)  # unused by recognize (reads pixels)

    tw = time.perf_counter()
    for i, c in enumerate(crops):
        if warm_progress:
            print(f"    warming/compiling crop {i + 1}/{len(crops)} ({c.width}x{c.height})...", flush=True)
        with silenced():
            rec.recognize(c, _region(c), opts)
    warm_s = time.perf_counter() - tw

    ms, texts = [], []
    for c in crops:
        t0 = time.perf_counter()
        txt = rec.recognize(c, _region(c), opts)
        ms.append((time.perf_counter() - t0) * 1000)
        texts.append(txt)
    rate = len(crops) / (sum(ms) / 1000)
    print(f"  [{label}] warm {warm_s:.1f}s | {rate:.3f} crops/sec | "
          f"per-crop ms min {min(ms):.0f} / med {statistics.median(ms):.0f} / max {max(ms):.0f}")
    return rate, ms, texts, warm_s


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # live progress under a Docker pipe

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default=os.getenv("BENCH_DATA"),
                    help="folder of pages (with --detect) or pre-cut crops; or $BENCH_DATA")
    ap.add_argument("--detect", action="store_true", help="treat the folder as pages: detect + deskew crops")
    ap.add_argument("--items", type=int, default=8, help="crops to compile+time (first N; each is a distinct compile)")
    ap.add_argument("--probe-cap", type=int, default=64, help="max_new_tokens per recognize")
    ap.add_argument("--mode", default="reduce-overhead",
                    choices=["reduce-overhead", "default", "max-autotune"],
                    help="torch.compile mode (reduce-overhead = HIP graphs, targets the launch overhead)")
    ap.add_argument("--dynamic", action="store_true",
                    help="torch.compile(dynamic=True): one graph for varying shapes (no HIP graphs, but no recompile)")
    args = ap.parse_args()
    if not args.data:
        sys.exit("no data path: pass a folder/image or set $BENCH_DATA")

    device, reason = paddle_device(False)
    if device is None:
        sys.exit(f"PaddleOCR-VL compile probe needs a GPU: {reason}")

    crops, src = load_crops(args.data, args.detect)
    crops = crops[:max(1, args.items)]
    opts = {"max_new_tokens": args.probe_cap}  # plugin fills max_pixels=150k/pow2 from OPTION_SCHEMA
    print(f"device {device} | {len(crops)} crops ({src}) | plugin cap 150k/pow2 | probe-cap {args.probe_cap}")

    print("\n== baseline (no compile) ==")
    rec = load_paddle(device, None)
    base_rate, base_ms, base_txt, _ = _pass(rec, crops, opts, "baseline")

    print(f"\n== compiled (static cache + torch.compile mode={args.mode} dynamic={args.dynamic}) ==")
    print("   (inductor compile is slow -- several minutes for N distinct shapes is normal)")
    try:
        import torch

        rec._model.generation_config.cache_implementation = "static"  # fixed-shape decode -> reusable graph
        rec._model.forward = torch.compile(
            rec._model.forward, mode=args.mode, fullgraph=False, dynamic=(True if args.dynamic else None))
        comp_rate, comp_ms, comp_txt, warm_s = _pass(rec, crops, opts, "compiled", warm_progress=True)
    except Exception:  # noqa: BLE001 - a compile/runtime failure IS the finding, not an error to hide
        import traceback
        print("\n  COMPILE/RUN FAILED -- this is the finding (compile doesn't survive on this stack):\n")
        traceback.print_exc()
        return 0

    print("\n== verdict ==")
    print(f"  {base_rate:.3f} -> {comp_rate:.3f} crops/sec = {comp_rate / base_rate:.2f}x   "
          f"(per-crop med {statistics.median(base_ms):.0f} -> {statistics.median(comp_ms):.0f} ms; "
          f"compile warm cost {warm_s:.0f}s for {len(crops)} shapes)")
    mismatch = [i for i in range(len(crops)) if base_txt[i] != comp_txt[i]]
    if mismatch:
        print(f"  !! OUTPUT CHANGED on {len(mismatch)}/{len(crops)} crops -- compile is NOT correctness-safe:")
        for i in mismatch[:6]:
            print(f"     #{i} base {base_txt[i]!r}")
            print(f"         comp {comp_txt[i]!r}")
    else:
        print(f"  outputs identical on all {len(crops)} crops (correctness-safe)")
    print("\n  NOTE: best case -- the timed pass hits already-compiled shapes. Production feeds a "
          "FRESH shape per crop (dynamic resolution), so a speedup here only transfers if that "
          f"~{warm_s / len(crops):.0f}s/shape recompile is acceptable (or --dynamic avoids it at the "
          "cost of the HIP-graph win).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
