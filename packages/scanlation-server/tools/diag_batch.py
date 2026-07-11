#!/usr/bin/env python3
"""Diagnose WHY PaddleOCR-VL batch recognize explodes vs per-crop.

The pipeline batch (one page's crops in one generate) ran 5.5x SLOWER than the
per-crop loop, and worse the more regions a page has (2 crops ~8x, 3 crops ~10x,
4 crops ~19x) — not the doc's 2.04x. The doc's bench reused ONE crop B times
(identical size AND output length -> zero padding, zero straggler); a real page
has ragged crops. The ~N^2 blow-up is the tell that something is quadratic.

This runs real crops per-crop, then as one batch, printing every input tensor's
SHAPE + per-crop-vs-batch timing, so we can tell which it is:
  (a) processor doesn't really batch -> input_ids is (1, huge), all images fused
      into ONE sequence (vision attention then O((N*tokens)^2))
  (b) vision-token padding -> pixel_values / grid balloon to the largest crop x N
  (c) straggler only -> input_ids is a clean (N, L), blow-up is ~N not ~N^2

Run it where the weights live (GPU box); PaddleOCR-VL only, ~60s/crop on CPU.
Point it at a pages folder (detect+deskew) or a single page for that page's crops:

    python tools/diag_batch.py PAGES_DIR --n 4
    python tools/diag_batch.py 19.jpg          # just that page's crops
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 - makes scanlation_*/app importable + UTF-8 stdio

import argparse
import sys
import time

from _bench_common import load_crops, load_paddle, paddle_device


def _shapes(inp) -> str:
    """One-line 'key(shape) key(shape) ...' of a processor's BatchFeature — the shapes
    are the whole point (is input_ids (N, L) or (1, huge)? how big is pixel_values?)."""
    parts = []
    for k, v in inp.items():
        shape = getattr(v, "shape", None)
        parts.append(f"{k}{tuple(shape)}" if shape is not None else f"{k}=?")
    return "  ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="pages folder (detect+deskew) or a single page/crop image")
    ap.add_argument("--n", type=int, default=4, help="how many crops to batch (default 4)")
    ap.add_argument("--no-detect", action="store_true",
                    help="data is already-cut crops, skip detect")
    ap.add_argument("--paddle-cpu", action="store_true", help="run on CPU anyway (~60s/crop)")
    args = ap.parse_args()

    device, reason = paddle_device(args.paddle_cpu)
    if device is None:
        sys.exit(f"PaddleOCR-VL unavailable: {reason}")

    from scanlation_sdk.local_engine import downscale_to_cap, to_rgb

    crops, source = load_crops(args.data, use_detect=not args.no_detect)
    sub = crops[: args.n]
    if len(sub) < 2:
        sys.exit(f"need >=2 crops to compare a batch; got {len(sub)} from {source}")
    print(f"{len(sub)} crops from {source} (of {len(crops)} total), device {device}\n")

    rec = load_paddle(device, None)
    opts = rec.resolve_options({})
    mp, mnt, mode = opts["max_pixels"], opts["max_new_tokens"], opts["downscale_mode"]
    proc, model = rec._proc, rec._model
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": rec.PROMPT}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    capped = [downscale_to_cap(to_rgb(c), mp, mode) for c in sub]

    # Warmup: the process's first forward JIT-compiles kernels (10-30s). Excluded
    # from every number below so the comparison isn't cold-start noise.
    _ = model.generate(
        **proc(text=[prompt], images=[capped[0]], return_tensors="pt").to(model.device),
        max_new_tokens=8, do_sample=False)

    print("== per-crop ==")
    pc_total = 0.0
    for i, c in enumerate(capped):
        inp = proc(text=[prompt], images=[c], return_tensors="pt").to(model.device)
        t = time.perf_counter()
        out = model.generate(**inp, max_new_tokens=mnt, do_sample=False)
        dt = (time.perf_counter() - t) * 1000
        ntok = int(out.shape[1] - inp["input_ids"].shape[1])
        print(f"  #{i} {c.size[0]}x{c.size[1]}px  {_shapes(inp)}  out={ntok}tok  {dt:.0f}ms")
        pc_total += dt

    print("\n== batch (all crops in ONE generate) ==")
    proc.tokenizer.padding_side = "left"
    inp = proc(text=[prompt] * len(capped), images=capped, padding=True, return_tensors="pt").to(model.device)
    print(f"  {_shapes(inp)}")
    t = time.perf_counter()
    out = model.generate(**inp, max_new_tokens=mnt, do_sample=False)
    dt = (time.perf_counter() - t) * 1000
    ntok = int(out.shape[1] - inp["input_ids"].shape[1])
    print(f"  out{tuple(out.shape)}  gen={ntok}tok  {dt:.0f}ms")

    n = len(capped)
    print("\n== 분석 ==")
    print(f"  per-crop 합 {pc_total:.0f}ms  vs  배치 {dt:.0f}ms  =  {dt / pc_total:.1f}x"
          f"{'  (배치가 느림)' if dt > pc_total else ''}")
    print(f"  input_ids 가 ({n}, L) 이면 진짜 배치 — L 이 per-crop 대비 얼마나 부푸나(패딩)를 본다.")
    print(f"  input_ids 가 (1, 큰값) 이면 프로세서가 배치를 안 하고 {n}장을 한 시퀀스로 뭉친 것.")
    print(f"  배율이 ~{n}x면 straggler(최장 출력이 배치 전체를 끌고 감), ~{n * n}x면 vision O(n^2)/시퀀스 폭발.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
