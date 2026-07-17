#!/usr/bin/env python3
"""Translate-only concurrency bench with saturated supply.

The pipeline sweep (``run_report --concurrency``) measures END-TO-END throughput,
where translate slots are fed by a serialized CPU recognize — it cannot answer
"what does translate concurrency ALONE buy?". This bench isolates that: it takes
the REAL page texts from a run_report markdown (each page = one batch body,
exactly what the llama.cpp plugin sends: same prompt builders, schema, sampling
incl. dry_multiplier), and fires them straight at llama-server with P in flight
at a time — as if an already-recognized backlog were feeding translate with zero
wait. Varied real prompts, so prefill is honestly paid (unlike an
identical-body curl loop, where the slot cache makes prefill ~free).

    python tools/bench_translate_concurrency.py run_report_20260716_004042.md
    python tools/bench_translate_concurrency.py report.md -P 1,2,4 --repeat 2

A warmup pass runs first so llama's slot prompt cache is equally warm for every
timed pass. ``max_tokens`` bounds each request (probe hygiene; normal outputs
are ~30-90 tokens so a 400 cap doesn't distort). Stdlib-only; run where
llama-server is reachable (--endpoint / LLAMACPP_ENDPOINT).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 - UTF-8 stdio (Japanese/Korean output on cp949 terminals)

import argparse
import json
import os
import re
import sys
import time
import types
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# prompt.py (and the context.py it pulls) are stdlib-only, but the package __init__
# is not — it re-exports the whole SDK, dragging numpy in. Register a stub package
# that skips __init__.py (same trick as diag_runaway.py) so this stays runnable on
# a bare GPU host.
_sdk_pkg = types.ModuleType("scanlation_sdk")
_sdk_pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "scanlation-sdk" / "scanlation_sdk")]
sys.modules.setdefault("scanlation_sdk", _sdk_pkg)
from scanlation_sdk.prompt import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    batch_schema,
    build_batch_prompt,
)

_PAGE = re.compile(r"^### (\S+)")
_ROW = re.compile(r"^\|\s*\d+\s*\|")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")  # cell separator, honoring run_report's \| escape


def _unescape(cell: str) -> str:
    """Reverse run_report._md_cell (pipes/backslashes; ' ⏎ ' was a newline)."""
    return cell.strip().replace(" ⏎ ", "\n").replace("\\|", "|").replace("\\\\", "\\")


def parse_pages(md_path: Path) -> list[tuple[str, list[str]]]:
    """(page name, non-blank source texts) per page, from a run_report markdown.
    Row shape: | # | [bounds] | source (OCR) | destination |"""
    pages: list[tuple[str, list[str]]] = []
    name, texts = None, []
    def flush():
        if name is not None and any(t.strip() for t in texts):
            pages.append((name, [t for t in texts if t.strip()]))
    for line in md_path.read_text(encoding="utf-8").splitlines():
        m = _PAGE.match(line)
        if m:
            flush()
            name, texts = m.group(1), []
            continue
        if name is not None and _ROW.match(line):
            cells = _CELL_SPLIT.split(line)
            if len(cells) >= 5:  # ['', ' # ', ' [bounds] ', ' source ', ' dest ', ...]
                texts.append(_unescape(cells[3]))
    flush()
    return pages


def _body(texts: list[str], args: argparse.Namespace) -> dict:
    """The llama.cpp plugin's batch body (plugin._body + _translate_batch_call),
    bounded with max_tokens for bench hygiene."""
    return {
        "model": args.model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": build_batch_prompt(texts, args.src, args.dst)},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "dry_multiplier": 0.8,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "translations", "schema": batch_schema(len(texts)), "strict": True},
        },
        "max_tokens": args.max_tokens,
    }


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _run_pass(url: str, bodies: list[dict], p: int, timeout: float) -> tuple[float, int]:
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=p) as ex:
        results = list(ex.map(lambda b: _post(url, b, timeout), bodies))
    wall = time.monotonic() - t0
    toks = sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in results)
    return wall, toks


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", help="run_report markdown with per-page source tables (e.g. a --no-translate run)")
    ap.add_argument("-P", "--concurrency", default="1,2,4",
                    help="comma-separated in-flight counts to sweep (default 1,2,4)")
    ap.add_argument("--repeat", type=int, default=1, help="timed passes per P (default 1)")
    ap.add_argument("--src", default="ja")
    ap.add_argument("--dst", default="ko")
    ap.add_argument("--model", default="", help="model id (llama-server ignores it)")
    ap.add_argument("--endpoint", default=os.getenv("LLAMACPP_ENDPOINT", "http://127.0.0.1:8080"))
    ap.add_argument("--max-tokens", type=int, default=400, help="per-request bound (bench hygiene)")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--no-warmup", action="store_true", help="skip the untimed cache-warming pass")
    ap.add_argument("--dry-run", action="store_true", help="parse the report and print pages, no requests")
    args = ap.parse_args()

    pages = parse_pages(Path(args.report))
    if not pages:
        sys.exit(f"no page tables found in {args.report}")
    bodies = [_body(texts, args) for _, texts in pages]
    n_texts = sum(len(t) for _, t in pages)
    print(f"{len(bodies)} pages / {n_texts} texts from {args.report}")
    if args.dry_run:
        for name, texts in pages:
            print(f"  {name}: {len(texts)} text(s)  {texts[0][:30]!r}...")
        return 0

    url = f"{args.endpoint.rstrip('/')}/v1/chat/completions"
    plist = [int(x) for x in args.concurrency.split(",")]
    if not args.no_warmup:
        # Warm at the highest P being measured: a lower-P warmup can leave some of
        # llama's slots cold, taxing the first requests of a higher-P timed pass
        # with the full prompt-prefix prefill (biases high P down).
        wall, _ = _run_pass(url, bodies, max(plist), args.timeout)
        print(f"warmup: P={max(plist)} {wall:.2f}s (untimed — equalizes slot cache + pre-heat)")
    for p in plist:
        for r in range(args.repeat):
            wall, toks = _run_pass(url, bodies, p, args.timeout)
            tag = f" (run {r + 1}/{args.repeat})" if args.repeat > 1 else ""
            print(f"P={p}{tag}: {len(bodies)} req in {wall:.2f}s  ->  "
                  f"{len(bodies) / wall:.2f} req/s  ·  {wall / len(bodies) * 1000:.0f}ms/req  ·  "
                  f"{toks} tok ({toks / wall:.1f} tok/s aggregate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
