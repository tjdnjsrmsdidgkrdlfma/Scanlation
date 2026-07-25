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
import html
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


def _parse_endpoints(spec: str) -> list[tuple[str, str]]:
    """``"a=http://x,http://y:8091"`` -> [("a", "http://x"), (":8091", "http://y:8091")].
    An unlabelled entry is named by its port, which is what actually distinguishes two
    servers that differ only in config."""
    out = []
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        label, _, url = part.rpartition("=")
        url = url.rstrip("/")
        out.append((label or ":" + url.rsplit(":", 1)[-1], url))
    return out


def _write_html(path: str, crops, ref_text, variants, src: str) -> None:
    """A crop-by-crop page: the IMAGE the models were given, then every variant's text.
    This exists because char-sim can only say outputs differ — deciding which is RIGHT
    needs the picture, since the reference is another model's guess. Rows where the
    variants disagree with the reference are flagged; identical rows are dimmed."""
    esc = html.escape
    parts = [f"<!doctype html><meta charset='utf-8'><title>recognize compare</title>",
             "<style>body{font:14px/1.5 system-ui;margin:2rem;max-width:1400px}"
             "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:.5rem;"
             "vertical-align:top;text-align:left}img{max-width:320px;height:auto;image-rendering:pixelated}"
             "pre{margin:0;white-space:pre-wrap;font:13px/1.45 ui-monospace,monospace}"
             "tr.same{opacity:.45}tr.diff td{background:#fff8e1}"
             "@media(prefers-color-scheme:dark){body{background:#111;color:#eee}td,th{border-color:#444}"
             "tr.diff td{background:#2a2410}}</style>",
             f"<h1>recognize compare</h1><p>{esc(src)} — {len(crops)} crops, cap {CAP_PIXELS}/{CAP_MODE}. "
             "Rows highlighted where a variant disagrees with the reference; the reference is itself a "
             "model output, so a disagreement is not automatically an error.</p>",
             "<table><tr><th>#</th><th>crop (as sent)</th><th>transformers (reference)</th>"
             + "".join(f"<th>{esc(l)}</th>" for l, _, _ in variants) + "</tr>"]
    for i, crop in enumerate(crops):
        texts = [v[1][i] for v in variants]
        ref = ref_text[i] if ref_text else ""
        cls = "diff" if ref_text and any(t != ref for t in texts) else "same"
        parts.append(f"<tr class='{cls}'><td>{i}</td>"
                     f"<td><img src='data:image/png;base64,{_png_b64(crop)}'><br>"
                     f"<small>{crop.width}×{crop.height} = {crop.width * crop.height // 1000}k px</small></td>"
                     f"<td><pre>{esc(ref)}</pre></td>"
                     + "".join(f"<td><pre>{esc(t)}</pre></td>" for t in texts) + "</tr>")
    parts.append("</table>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


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
                    help="llama-server base URL serving the PaddleOCR-VL GGUF (+ its mmproj). Comma-separate "
                         "several (optionally 'label=url') to compare configs side by side in ONE run — e.g. two "
                         "servers differing only in the mmproj's image_min_pixels.")
    ap.add_argument("--html", default="",
                    help="also write an HTML report embedding each CROP IMAGE next to every variant's text. "
                         "char-sim only says outputs DIFFER, not which is right — the reference is another "
                         "model's guess, not ground truth — so judging needs the picture.")
    ap.add_argument("--items", type=int, default=12, help="crops to time")
    ap.add_argument("--max-tokens", type=int, default=256, help="output cap per crop (both passes)")
    ap.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout per crop, seconds")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the transformers pass (endpoint smoke test only; no accuracy comparison)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="in-flight requests for the llama.cpp pass (its server has n_slots; >1 measures "
                         "whether those slots add throughput, the HTTP analog of the process pool's W). "
                         "Rate is then wall-clock over the whole batch, not the sum of per-crop times.")
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

    # --- candidate passes: one per endpoint ----------------------------------
    variants = []  # (label, texts, rate)
    for label, url in _parse_endpoints(args.endpoint):
        print(f"\n== candidate: {label} ({url}) ==")
        try:  # one untimed warm call: absorbs model/mmproj load if the server is cold
            with silenced():
                llamacpp_recognize(url, capped[0], args.max_tokens, args.timeout)
        except urllib.error.URLError as exc:
            sys.exit(f"cannot reach llama-server at {url}: {exc}\n"
                     "Start it with the PaddleOCR-VL GGUF + --mmproj (needs llama.cpp build b8110+).")
        except Exception as exc:  # noqa: BLE001 - a bad response IS the finding
            sys.exit(f"llama-server rejected the request: {type(exc).__name__}: {exc}")

        def _one(c, _url=url):
            t0 = time.perf_counter()
            text = llamacpp_recognize(_url, c, args.max_tokens, args.timeout)
            return text, (time.perf_counter() - t0) * 1000

        t_batch = time.perf_counter()
        if args.concurrency > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                out = list(ex.map(_one, capped))  # order preserved
            got, ms = [r[0] for r in out], [r[1] for r in out]
        else:
            got, ms = [], []
            for i, c in enumerate(capped):
                text, el = _one(c)
                got.append(text)
                ms.append(el)
                print(f"    {i:>3} {c.width:>4}x{c.height:<4} {el:>7.0f}ms  {text[:40]!r}")
        batch_s = time.perf_counter() - t_batch

        if args.concurrency > 1:
            # Concurrent: per-crop times overlap, so the sum is meaningless -- the real
            # throughput is the batch wall clock (same convention as the K/W gate bench).
            rate = len(capped) / batch_s
            print(f"  [{label} c={args.concurrency}] {rate:.3f} crops/sec (batch wall {batch_s:.1f}s) | "
                  f"per-crop ms min {min(ms):.0f} / med {statistics.median(ms):.0f} / max {max(ms):.0f}")
            rows.append(f"| {label} (concurrency {args.concurrency}) | {rate:.3f} | {min(ms):.0f} | "
                        f"{statistics.median(ms):.0f} | {max(ms):.0f} |")
        else:
            rate = _report_pass(label, ms, rows)
        variants.append((label, got, rate))
    rows.append("")

    # --- verdict -------------------------------------------------------------
    print("\n== verdict ==")
    for label, got, rate in variants:
        if ref_rate:
            exact = sum(a == b for a, b in zip(ref_text, got))
            sim = sum(difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(ref_text, got)) / len(got)
            empty = sum(1 for g in got if not g)
            print(f"  [{label}] speed {rate / ref_rate:.2f}x | vs transformers: exact {exact}/{len(got)}, "
                  f"char-sim {sim:.3f}" + (f" | !! {empty} EMPTY" if empty else ""))
            rows.append(f"- **{label}**: {rate / ref_rate:.2f}x speed; vs transformers exact {exact}/{len(got)}, "
                        f"char-sim {sim:.3f}" + (f", {empty} EMPTY" if empty else ""))
        else:
            print(f"  [{label}] {rate:.3f} crops/sec (no reference pass -- accuracy NOT checked)")
            rows.append(f"- **{label}**: {rate:.3f} crops/sec, no reference pass")
    rows.append("")
    if ref_rate:
        # char-sim ranks agreement with the reference, NOT correctness -- the reference is
        # itself a model's guess. Print every disagreeing crop so a human can judge, and
        # point at the HTML (which carries the picture) when one was written.
        rows += ["> char-sim measures agreement with the transformers reference, which is itself a model "
                 "output, not ground truth. A lower number can mean the variant is BETTER. Judge the rows "
                 "below against the crop image.", "",
                 "| # | transformers | " + " | ".join(l for l, _, _ in variants) + " |",
                 "|---|---|" + "---|" * len(variants)]
        for i in range(len(capped)):
            texts = [v[1][i] for v in variants]
            if any(t != ref_text[i] for t in texts):
                print(f"    #{i} ref {ref_text[i]!r}")
                for (label, got, _), t in zip(variants, texts):
                    print(f"        {label:>12} {t!r}")
                rows.append(f"| {i} | `{ref_text[i]}` | " + " | ".join(f"`{t}`" for t in texts) + " |")

    if args.html:
        _write_html(args.html, capped, ref_text, variants, src)
        print(f"\nHTML (crop images + every variant): {args.html}")

    return write_report(rows, "bench_report_llamacpp")


if __name__ == "__main__":
    raise SystemExit(main())
