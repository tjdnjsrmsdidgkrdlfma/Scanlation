#!/usr/bin/env python3
"""Probe: can llama.cpp serve PaddleOCR-VL faster than transformers, and read the same?

recognize-decode-bound.md found per-crop cost is ~95% decode at a FLAT ~64ms/token
while weight-read is ~9% and compute <1% -- i.e. the time is HOST-side overhead of the
eager transformers loop, not GPU work. torch.compile can't remove it here (Triton/
compiler absent; graph capture incompatible with dynamic resolution). llama.cpp has no
Python/torch dispatch layer at all, so it sidesteps that whole class -- and upstream
merged PaddleOCR-VL support (ggml-org/llama.cpp#18825, build b8110+), with a GGUF of
our exact fine-tune published at adambarbato/PaddleOCR-VL-For-Manga-GGUF.

Two things must both hold for the switch to be worth it, so this measures BOTH:

  speed     per-crop ms / crops-per-sec vs the transformers reference
  accuracy  is the text the SAME? llama.cpp's maintainer flagged PaddleOCR-VL as
            "may have degraded performance" with no benchmark -- this is that
            benchmark, on our own manga crops.

    python tools/bench_recognize_llamacpp.py PAGES_DIR --detect --endpoint http://host.docker.internal:8090

Run it in the container (the reference pass needs torch + the installed plugin); the
llama-server can live on the host (extra_hosts maps host.docker.internal). Both passes
see the SAME crops with the SAME production cap (150k/pow2) applied, so the comparison
is apples-to-apples. Use --no-reference for a quick endpoint smoke test on its own.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 - makes `app`/`scanlation_*` importable + UTF-8 stdio

import argparse
import base64
import difflib
import io
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

from _bench_common import load_crops, load_paddle, paddle_device, silenced, write_report

# Production defaults, mirrored from the PaddleOCR-VL plugin's OPTION_SCHEMA so both
# passes downscale identically. The transformers pass applies these INSIDE recognize();
# the llama.cpp pass has no plugin, so this script applies them to the crop it uploads.
CAP_PIXELS = int(os.environ.get("SCANLATION_RECOGNIZE_MAX_PIXELS", "150000"))
CAP_MODE = os.environ.get("SCANLATION_RECOGNIZE_DOWNSCALE_MODE", "pow2")
PROMPT = "OCR:"  # the plugin's prompt


def _png_b64(crop) -> str:
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _post(endpoint: str, path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def llamacpp_recognize(endpoint: str, crop, max_tokens: int, timeout: float) -> str:
    """One crop -> text via llama-server's OpenAI-compatible vision chat endpoint.
    temperature 0 = greedy, matching the plugin's deterministic default."""
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_png_b64(crop)}"}},
            {"type": "text", "text": PROMPT},
        ]}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = _post(endpoint, "/v1/chat/completions", body, timeout)
    return (data["choices"][0]["message"]["content"] or "").strip()


def _report_pass(label: str, ms: list[float], rows: list) -> float:
    rate = len(ms) / (sum(ms) / 1000)
    print(f"  [{label}] {rate:.3f} crops/sec | per-crop ms "
          f"min {min(ms):.0f} / med {statistics.median(ms):.0f} / max {max(ms):.0f}")
    rows.append(f"| {label} | {rate:.3f} | {min(ms):.0f} | {statistics.median(ms):.0f} | {max(ms):.0f} |")
    return rate


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # live progress under a Docker pipe

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default=os.getenv("BENCH_DATA"),
                    help="folder of pages (with --detect) or pre-cut crops; or $BENCH_DATA")
    ap.add_argument("--detect", action="store_true", help="treat the folder as pages: detect + deskew crops")
    ap.add_argument("--endpoint", default=os.getenv("LLAMACPP_RECOGNIZE_ENDPOINT", "http://host.docker.internal:8090"),
                    help="llama-server base URL serving the PaddleOCR-VL GGUF (+ its mmproj)")
    ap.add_argument("--items", type=int, default=12, help="crops to time")
    ap.add_argument("--max-tokens", type=int, default=256, help="output cap per crop (both passes)")
    ap.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout per crop, seconds")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the transformers pass (endpoint smoke test only; no accuracy comparison)")
    args = ap.parse_args()
    if not args.data:
        sys.exit("no data path: pass a folder/image or set $BENCH_DATA")

    crops, src = load_crops(args.data, args.detect)
    crops = crops[:max(1, args.items)]

    # Apply the production cap ONCE, here, to the crops the llama.cpp pass uploads. The
    # transformers pass re-applies the same cap inside recognize() -- idempotent, since a
    # crop already under the cap is returned unchanged.
    from scanlation_sdk.local_engine import downscale_to_cap, to_rgb
    capped = [downscale_to_cap(to_rgb(c), CAP_PIXELS, CAP_MODE) for c in crops]
    n_down = sum(1 for a, b in zip(crops, capped) if a is not b)
    print(f"{len(crops)} crops ({src}) | cap {CAP_PIXELS}/{CAP_MODE}: {n_down} downscaled | "
          f"max_tokens {args.max_tokens}")
    print(f"endpoint: {args.endpoint}")

    rows = ["# recognize: llama.cpp vs transformers", "",
            f"- crops: {len(crops)} ({src}), cap {CAP_PIXELS}/{CAP_MODE}",
            f"- endpoint: {args.endpoint}", "",
            "| pass | crops/sec | min ms | med ms | max ms |", "|---|---|---|---|---|"]

    # --- reference pass: the production transformers path -------------------
    ref_text, ref_rate = None, None
    if not args.no_reference:
        device, reason = paddle_device(False)
        if device is None:
            sys.exit(f"reference pass needs a GPU: {reason} (use --no-reference to skip)")
        print(f"\n== reference: transformers on {device} ==")
        from scanlation_sdk.contracts import Region
        rec = load_paddle(device, None)
        opts = {"max_new_tokens": args.max_tokens}
        with silenced():
            rec.recognize(capped[0], Region.from_bbox(0, 0, capped[0].width, capped[0].height), opts)  # warm JIT
        ref_text, ref_ms = [], []
        for c in capped:
            t0 = time.perf_counter()
            ref_text.append(rec.recognize(c, Region.from_bbox(0, 0, c.width, c.height), opts))
            ref_ms.append((time.perf_counter() - t0) * 1000)
        ref_rate = _report_pass("transformers", ref_ms, rows)

    # --- candidate pass: llama.cpp ------------------------------------------
    print(f"\n== candidate: llama.cpp ==")
    try:  # one untimed warm call: absorbs model/mmproj load if the server is cold
        with silenced():
            llamacpp_recognize(args.endpoint, capped[0], args.max_tokens, args.timeout)
    except urllib.error.URLError as exc:
        sys.exit(f"cannot reach llama-server at {args.endpoint}: {exc}\n"
                 "Start it with the PaddleOCR-VL GGUF + --mmproj (needs llama.cpp build b8110+).")
    except Exception as exc:  # noqa: BLE001 - a bad response IS the finding
        sys.exit(f"llama-server rejected the request: {type(exc).__name__}: {exc}")

    got, ms = [], []
    for i, c in enumerate(capped):
        t0 = time.perf_counter()
        got.append(llamacpp_recognize(args.endpoint, c, args.max_tokens, args.timeout))
        ms.append((time.perf_counter() - t0) * 1000)
        print(f"    {i:>3} {c.width:>4}x{c.height:<4} {ms[-1]:>7.0f}ms  {got[-1][:40]!r}")
    cand_rate = _report_pass("llama.cpp", ms, rows)
    rows.append("")

    # --- verdict -------------------------------------------------------------
    print("\n== verdict ==")
    if ref_rate:
        print(f"  speed: {ref_rate:.3f} -> {cand_rate:.3f} crops/sec = {cand_rate / ref_rate:.2f}x")
        rows += [f"**speed: {cand_rate / ref_rate:.2f}x** ({ref_rate:.3f} -> {cand_rate:.3f} crops/sec)", ""]
        exact = sum(a == b for a, b in zip(ref_text, got))
        sim = sum(difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(ref_text, got)) / len(got)
        empty = sum(1 for g in got if not g)
        print(f"  accuracy vs transformers: exact {exact}/{len(got)} | char-sim {sim:.3f}"
              + (f" | !! {empty} EMPTY outputs" if empty else ""))
        rows += [f"accuracy vs transformers: exact {exact}/{len(got)}, char-sim {sim:.3f}"
                 + (f", {empty} EMPTY" if empty else ""), "",
                 "| # | transformers | llama.cpp |", "|---|---|---|"]
        for i, (a, b) in enumerate(zip(ref_text, got)):
            if a != b:
                print(f"    #{i} ref {a!r}")
                print(f"        got {b!r}")
                rows.append(f"| {i} | `{a}` | `{b}` |")
    else:
        print(f"  {cand_rate:.3f} crops/sec (no reference pass -- accuracy NOT checked)")
        rows += [f"{cand_rate:.3f} crops/sec, no reference pass", ""]

    return write_report(rows, "bench_report_llamacpp")


if __name__ == "__main__":
    raise SystemExit(main())
