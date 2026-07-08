#!/usr/bin/env python3
"""Benchmark: recognize crop-batching -- does stacking a page's crops into one
forward pay off, and by how much?

Point it at a folder of real manga pages and cut crops the way the pipeline does:

    python tools/bench_recognize_batch.py PAGES_DIR --detect

(or a folder of already-cut crops without --detect; $BENCH_DATA sets the path).
It sweeps batch sizes 1/2/4/8/16, prints tables, and writes a report file
(bench_report_batch_<time>.md) next to where you ran it.

It answers the open number in tools/recognize-crop-batching.md -- c_B/c_1, the
per-step cost inflation at batch B (the doc guessed c8 ~ 2*c1) -- for the two
recognizers, each in the device it actually deploys on:

  manga-ocr (CPU): three measurements per batch size --
    * encoder-only  : the ViT forward (non-autoregressive, straggler-IMMUNE) ->
                      the clean batch-scaling ceiling, the "encoder bank".
    * fixed-length  : full generate forced to a fixed decode length -> the
                      straggler-FREE whole-model c_B/c_1 (the headline number).
    * natural       : real generate over the crop set + the output-length
                      distribution -> what straggler actually costs on this data.
    The batch path replicates MangaOcr.__call__'s grayscale + post_process, so
    it reads the same as the per-crop path (bypassing __call__ would silently
    drop the .convert("L") and regress accuracy).

  PaddleOCR-VL (GPU): per-crop throughput baseline (~1s/crop check), then a
    batch PROBE -- feed processor(text=[..], images=[..], padding=True) and
    compare the batched output to the per-crop output. The point is correctness:
    dynamic-resolution ragged vision tokens are the "silently wrong" kind, so a
    matching output is the gate, a speedup number is the bonus.

Needs the engines' weights installed, so run it where the models live. Paddle
runs only if a CUDA GPU is present (CPU is ~60s/crop -> skipped; --paddle-cpu to
force). Options:
    PAGES_DIR        folder of manga pages (with --detect) or pre-cut crops; or $BENCH_DATA
    --detect         treat the folder as pages: detect + deskew real bubble crops
    --batch SPEC     batch-size sweep (default "1,2,4,8,16")
    --items N        manga-ocr crops timed per batch size (default 96)
    --fixed-len L    forced decode length for the straggler-free run (default 32)
    --no-manga       skip the manga-ocr half
    --no-paddle      skip the PaddleOCR-VL half
    --paddle-cpu     run PaddleOCR-VL on CPU anyway (slow; for a no-GPU box)
    --paddle-items N per-crop baseline crops for PaddleOCR-VL (default 24)
    --probe-cap N    max_new_tokens for the paddle batch probe (default 256)
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 - makes `scanlation_*`/`app` importable + UTF-8 stdio

import argparse
import contextlib
import os
import statistics
import sys
import time
from datetime import datetime

from _bench_common import load_crops, load_paddle, paddle_device, silenced, write_report


# --- timing helper -----------------------------------------------------------
def _sec_per_call(call, reps: int) -> float:
    """Warm once, then time `reps` calls; return mean seconds per call."""
    call()  # warmup (weights/kernels hot, first-call graph build excluded)
    t0 = time.perf_counter()
    for _ in range(reps):
        call()
    return (time.perf_counter() - t0) / reps


# --- manga-ocr (CPU) ---------------------------------------------------------
def bench_manga(crops, batch_sizes, items: int, fixed_len: int, rows: list) -> None:
    """manga-ocr batch scaling on CPU: encoder-only, fixed-length, natural."""
    import torch
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()  # silence per-call max_length/max_new_tokens warnings
    from manga_ocr import MangaOcr
    from manga_ocr.ocr import post_process

    with silenced():
        m = MangaOcr(force_cpu=True)
    model, proc, tok = m.model, m.processor, m.tokenizer
    model.eval()  # manga-ocr runs on CPU with torch's default thread count

    # __call__ does img.convert("L").convert("RGB"); the batch path must too, or
    # it silently feeds colour where the model expects desaturated input.
    gray = [c.convert("L").convert("RGB") for c in crops]

    def pv(n):  # (n, C, 224, 224) -- ViTImageProcessor resizes every crop to 224
        batch = [gray[i % len(gray)] for i in range(n)]
        return proc(batch, return_tensors="pt").pixel_values

    def _run(header, kind):
        """Sweep batch sizes for one measurement kind; return list of (B, sec)."""
        out = []
        for b in batch_sizes:
            x = pv(b)
            if kind == "encoder":
                def call(x=x):
                    with torch.no_grad():
                        model.encoder(pixel_values=x)
            else:  # fixed-length generate: forced L decode steps, no straggler
                def call(x=x):
                    with torch.no_grad():
                        model.generate(x, min_new_tokens=fixed_len, max_new_tokens=fixed_len,
                                       do_sample=False, num_beams=1)
            reps = max(3, round(items / b))
            sec = _sec_per_call(call, reps)
            out.append((b, sec))
        return out

    def _table(title, note, measured):
        base = measured[0][1] / measured[0][0]  # sec/crop at B=1
        lines = [f"### manga-ocr (CPU) -- {title}", "", note, "",
                 "| batch | ms/batch | crops/sec | c_B/c_1 | speedup |",
                 "|---|---|---|---|---|"]
        print(f"\n-- manga-ocr CPU: {title}")
        print(f"{'batch':>6} {'ms/batch':>10} {'crops/sec':>10} {'c_B/c_1':>8} {'speedup':>8}")
        t1 = measured[0][1]
        for b, sec in measured:
            rate = b / sec
            cb = sec / t1                 # per-batch-step cost vs a batch-1 step
            speedup = (base) / (sec / b)  # crops/sec vs B=1 crops/sec
            print(f"{b:>6} {sec * 1000:>10.1f} {rate:>10.2f} {cb:>8.2f} {speedup:>7.2f}x")
            lines.append(f"| {b} | {sec * 1000:.1f} | {rate:.2f} | {cb:.2f} | {speedup:.2f}x |")
        rows.extend(lines + [""])

    _table("encoder-only (straggler-immune ceiling)",
           "The ViT forward alone -- non-autoregressive, so no straggler. This is "
           "the batchable part that manga-ocr banks regardless of decode length.",
           _run("encoder", "encoder"))
    _table(f"fixed-length generate (L={fixed_len}, straggler-free)",
           f"Full model forced to exactly {fixed_len} decode steps per crop -> the "
           "clean whole-model c_B/c_1 with decode-length variance removed. "
           "**This is the headline c8/c1.**",
           _run("gen", "gen"))

    # natural generate: real output lengths over the ACTUAL crop set (no cycling)
    # -> the straggler distribution + the throughput straggler actually leaves.
    print("\n-- manga-ocr CPU: natural generate (real straggler)")
    lengths = []
    for c in gray:
        with torch.no_grad():
            out = model.generate(proc([c], return_tensors="pt").pixel_values, max_length=300)
        lengths.append(int(out.shape[1]) - 1)  # generated tokens (drop decoder-start)
    lengths.sort()
    p = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))]  # noqa: E731
    dist = (f"output tokens over {len(lengths)} crops: "
            f"min={lengths[0]} median={statistics.median(lengths):.0f} "
            f"p90={p(0.9)} max={lengths[-1]}")
    print("  " + dist)

    print(f"{'batch':>6} {'crops/sec':>10} {'speedup':>8}")
    nat, base = [], None
    for b in batch_sizes:
        batches = [gray[i:i + b] for i in range(0, len(gray), b)]
        def call(batches=batches):
            for grp in batches:
                with torch.no_grad():
                    model.generate(proc(grp, return_tensors="pt").pixel_values, max_length=300)
        sec = _sec_per_call(call, 1)
        rate = len(gray) / sec
        if base is None:
            base = rate
        nat.append((b, rate, rate / base))  # speedup vs B=1 -- the real batching gain
        print(f"{b:>6} {rate:>10.2f} {rate / base:>7.2f}x")

    # sanity: show that the batch path decodes to real text (grayscale+post_process)
    with torch.no_grad():
        sample_out = model.generate(pv(min(4, len(gray))), max_length=300)
    sample = [post_process(t) for t in tok.batch_decode(sample_out, skip_special_tokens=True)]
    print(f"  batch-path decode sanity: {sample!r}")

    rows += [
        "### manga-ocr (CPU) -- natural generate (real straggler)", "",
        f"- {dist}", "",
        "| batch | crops/sec | speedup |", "|---|---|---|",
        *[f"| {b} | {r:.2f} | {s:.2f}x |" for b, r, s in nat],
        "",
        f"- batch-path decode sanity (should read as text): {sample!r}",
        "",
    ]


# --- PaddleOCR-VL (GPU) ------------------------------------------------------
def bench_paddle(crops, batch_sizes, device: str, items: int, probe_cap: int, rows: list) -> None:
    """PaddleOCR-VL per-crop baseline + a batch correctness/scaling PROBE."""
    import torch
    from scanlation_sdk.contracts import Region

    # Dev probe: force an attention backend (e.g. BENCH_ATTN=sdpa) to see if the
    # ROCm flash/mem-efficient SDPA kernels move the per-crop number. Leave BENCH_ATTN
    # unset for the plugin's default (eager for this model).
    attn = os.getenv("BENCH_ATTN")
    if attn:
        print(f"\n[BENCH_ATTN] reloading model with attn_implementation={attn!r}")
    rec = load_paddle(device, attn)
    if attn:
        rows += [f"- attn_implementation forced to `{attn}` (BENCH_ATTN)", ""]
    region = Region.from_bbox(0, 0, crops[0].width, crops[0].height)  # unused by recognize
    opts = {"max_new_tokens": probe_cap}  # cap both ref + batch the same -> apples to apples

    # per-crop baseline (~1s/crop check)
    n = max(2, items)
    sample = [crops[i % len(crops)] for i in range(n)]
    rec.recognize(sample[0], region, opts)  # warmup
    t0 = time.perf_counter()
    ref = [rec.recognize(c, region, opts) for c in sample]
    dt = time.perf_counter() - t0
    base_rate = n / dt
    print(f"\n-- PaddleOCR-VL {device}: per-crop baseline")
    print(f"  {base_rate:.2f} crops/sec  ({dt / n * 1000:.0f} ms/crop over {n} crops)")
    rows += [f"### PaddleOCR-VL ({device}) -- per-crop baseline", "",
             f"- {base_rate:.2f} crops/sec ({dt / n * 1000:.0f} ms/crop over {n} crops)", ""]

    # batch probe: does the processor batch multi-image, and does it read the same?
    proc, model = rec._proc, rec._model
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": rec.PROMPT}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    with contextlib.suppress(Exception):
        proc.tokenizer.padding_side = "left"  # gen frontier aligns at the right edge

    print(f"\n-- PaddleOCR-VL {device}: batch probe (correctness gate)")
    print(f"{'batch':>6} {'ran':>5} {'match':>7} {'crops/sec':>10} {'speedup':>8}")
    probe_rows = ["### PaddleOCR-VL -- batch probe (실측 3)", "",
                  "Feed `processor(text=[..], images=[..], padding=True)` + batched "
                  "generate; compare each row to the per-crop output. Dynamic-res "
                  "ragged vision tokens are the 'silently wrong' kind, so **match is "
                  "the gate**.", "",
                  "| batch | ran | match | crops/sec | speedup |",
                  "|---|---|---|---|---|"]
    for b in batch_sizes:
        if b == 1:
            continue
        imgs = [crops[i % len(crops)] for i in range(b)]
        want = [rec.recognize(c, region, opts) for c in imgs]  # per-crop reference
        try:
            inputs = proc(text=[text] * b, images=list(imgs),
                          padding=True, return_tensors="pt").to(model.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=probe_cap, do_sample=False)
            sec = time.perf_counter() - t0
            gen = out[:, inputs["input_ids"].shape[1]:]  # left-pad -> uniform prompt len
            got = [proc.decode(g, skip_special_tokens=True).strip() for g in gen]
            matches = sum(a == b_ for a, b_ in zip(want, got))
            rate = b / sec
            speed = rate / base_rate
            ran, mtxt = "yes", f"{matches}/{b}"
            print(f"{b:>6} {ran:>5} {mtxt:>7} {rate:>10.2f} {speed:>7.2f}x")
            probe_rows.append(f"| {b} | yes | {matches}/{b} | {rate:.2f} | {speed:.2f}x |")
            if matches < b:  # mismatch: batching is not output-preserving here -> show what differs
                for j, (w, g) in enumerate(zip(want, got)):
                    if w != g:
                        print(f"         mismatch #{j}: want={w!r} got={g!r}")
                        probe_rows.append(f"| | | want[{j}]={w!r} got[{j}]={g!r} | | |")
        except Exception as exc:  # noqa: BLE001 - "processor can't batch multi-image" is a valid finding
            print(f"{b:>6} {'no':>5}   {type(exc).__name__}: {exc}")
            probe_rows.append(f"| {b} | no | {type(exc).__name__}: {exc} | | |")
            break  # larger B won't fare better
    rows += probe_rows + [""]


def main() -> int:
    # Line-buffer stdout so progress shows live even when it's a Docker pipe
    # (non-TTY defaults to block buffering -> output would only appear at the end).
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default=os.getenv("BENCH_DATA"),
                    help="folder of pages OR a single image (with --detect), or pre-cut crops; or set $BENCH_DATA")
    ap.add_argument("--detect", action="store_true",
                    help="treat the folder as pages: detect + deskew real bubble crops")
    ap.add_argument("--batch", default="1,2,4,8,16", help='batch-size sweep, e.g. "1,2,4,8,16"')
    ap.add_argument("--items", type=int, default=96, help="manga-ocr crops timed per batch size")
    ap.add_argument("--fixed-len", type=int, default=32, help="forced decode length (straggler-free run)")
    ap.add_argument("--no-manga", action="store_true", help="skip the manga-ocr half")
    ap.add_argument("--no-paddle", action="store_true", help="skip the PaddleOCR-VL half")
    ap.add_argument("--paddle-cpu", action="store_true", help="run PaddleOCR-VL on CPU anyway (slow)")
    ap.add_argument("--paddle-items", type=int, default=24, help="per-crop baseline crops for PaddleOCR-VL")
    ap.add_argument("--probe-cap", type=int, default=256, help="max_new_tokens for the paddle batch probe")
    args = ap.parse_args()
    if not args.data:
        ap.error("a folder of manga pages is required (add --detect to cut crops), or set $BENCH_DATA")

    batch_sizes = [int(x) for x in args.batch.split(",") if x.strip()]
    crops, source = load_crops(args.data, args.detect)

    rows = [
        "# recognize crop-batching benchmark", "",
        f"- when: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- crops: {len(crops)} ({source})",
        f"- batch sweep: {batch_sizes}",
        "",
        "c_B/c_1 = a batch-B step's cost vs a batch-1 step (the doc's guess: c8~2). "
        "speedup = crops/sec at B vs at B=1.",
        "",
    ]
    print("\n".join(rows[:5]))

    if not args.no_manga:
        bench_manga(crops, batch_sizes, args.items, args.fixed_len, rows)

    if not args.no_paddle:
        device, reason = paddle_device(args.paddle_cpu)
        if device is None:
            msg = f"PaddleOCR-VL skipped: {reason} (CPU is ~60s/crop; --paddle-cpu to force)."
            print(f"\n{msg}")
            rows += [f"_{msg}_", ""]
        else:
            try:
                bench_paddle(crops, batch_sizes, device,
                             args.paddle_items, args.probe_cap, rows)
            except Exception as exc:  # noqa: BLE001 - a missing paddle install must not lose the manga table
                print(f"\nPaddleOCR-VL failed: {type(exc).__name__}: {exc}")
                rows += [f"_PaddleOCR-VL failed: {type(exc).__name__}: {exc}_", ""]

    return write_report(rows, "bench_report_batch")


if __name__ == "__main__":
    raise SystemExit(main())
