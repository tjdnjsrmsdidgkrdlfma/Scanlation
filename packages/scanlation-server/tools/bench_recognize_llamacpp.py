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


def llamacpp_recognize(endpoint: str, crop, max_tokens: int, timeout: float,
                       cache_prompt: bool = False) -> str:
    """One crop -> text via llama-server's OpenAI-compatible vision chat endpoint.
    temperature 0 = greedy, matching the plugin's deterministic default.

    ``cache_prompt`` is OFF here on purpose. llama-server caches a prompt (image
    included) per slot, so a crop it has already seen returns in tens of ms instead of
    doing the ~740ms vision prefill — and since a bench re-sends byte-identical crops
    every run, that silently inflates the rate by several x. It bit this investigation
    twice before the timings gave it away, so the bench opts out rather than relying on
    remembering to restart the server."""
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_png_b64(crop)}"}},
            {"type": "text", "text": PROMPT},
        ]}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": cache_prompt,
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


def _read_html(path: str):
    """Recover ``(crops, ref_text, variants, src)`` from a page this tool wrote.

    The page already embeds everything it was built from — the crop PNGs and every
    variant's text — so restyling it must not cost another run: a 42-crop pass burns
    minutes of GPU and two live llama-servers, while the layout is the part we actually
    iterate on."""
    import re

    from PIL import Image

    doc = open(path, encoding="utf-8").read()
    labels = [re.sub(r"\s*\(reference\)$", "", h) for h in re.findall(r"<th>(.*?)</th>", doc, re.S)][2:]
    rows = []
    for block in re.findall(r"<tr class='\w+'>(.*?)</tr>", doc, re.S):
        idx = int(re.search(r"<td>(\d+)</td>", block).group(1))
        b64 = re.search(r"base64,([A-Za-z0-9+/=]+)'", block).group(1)
        crop = Image.open(io.BytesIO(base64.b64decode(b64)))
        crop.load()
        # <mark> is added by the renderer, not part of the text -> strip before reuse.
        texts = [html.unescape(re.sub(r"</?mark>", "", p))
                 for p in re.findall(r"<pre>(.*?)</pre>", block, re.S)]
        rows.append((idx, crop, texts))
    if not rows:
        sys.exit(f"{path}: no rows found — is this a page written by this tool?")
    rows.sort(key=lambda r: r[0])
    crops = [r[1] for r in rows]
    cols = [[r[2][k] for r in rows] for k in range(len(rows[0][2]))]
    m = re.search(r"<div>(.*?) — \d+ crops", doc, re.S)
    src = html.unescape(m.group(1)) if m else re.sub(r" — .*", "", path)
    return crops, cols[0], [(labels[k], cols[k], 0.0) for k in range(1, len(cols))], src


def _squash(s: str) -> str:
    """Text with ALL whitespace removed — the key for "same reading, different wrapping".
    Line breaks and spaces are the single most common disagreement between these
    variants and say nothing about who read the glyphs right, so rows that only differ
    that way must not compete for attention with real misreads."""
    return "".join(s.split())


def _marked(ref: str, s: str, esc) -> str:
    """``s`` with the runs that differ from ``ref`` wrapped in <mark> — the eye goes
    straight to the disputed glyphs instead of re-reading the whole line."""
    out = []
    for op, _i1, _i2, j1, j2 in difflib.SequenceMatcher(None, ref, s).get_opcodes():
        if j1 == j2:
            continue
        chunk = esc(s[j1:j2])
        out.append(chunk if op == "equal" else f"<mark>{chunk}</mark>")
    return "".join(out)


# Sibling of tools/compare/html.py's vote page (same .eng/data-eng/data-key +
# localStorage convention, so the two feel like one family). Kept standalone rather
# than importing it: that one is built around the compare harness's per-category
# aggregation, which this bench has no notion of.
_VOTE_JS = """
(function(){
  var K='llamacppsel:', tally={};
  function refresh(){
    document.getElementById('tally').innerHTML='선택수 — '+engs.map(function(e){
      return '<b>'+e+'</b> '+(tally[e]||0);}).join(' &nbsp;·&nbsp; ');
  }
  function set(td,on){
    var e=td.getAttribute('data-eng'), k=K+td.getAttribute('data-key');
    if(on){td.classList.add('sel');tally[e]=(tally[e]||0)+1;try{localStorage.setItem(k,'1')}catch(x){}}
    else {td.classList.remove('sel');tally[e]=(tally[e]||0)-1;try{localStorage.removeItem(k)}catch(x){}}
  }
  document.addEventListener('click',function(ev){
    var td=ev.target.closest&&ev.target.closest('.eng'); if(!td) return;
    set(td,!td.classList.contains('sel')); refresh();
  });
  document.querySelectorAll('.eng').forEach(function(td){
    var on=false; try{on=localStorage.getItem(K+td.getAttribute('data-key'))==='1'}catch(x){}
    if(on){td.classList.add('sel');var e=td.getAttribute('data-eng');tally[e]=(tally[e]||0)+1;}
  });
  document.getElementById('reset').addEventListener('click',function(){
    document.querySelectorAll('.eng.sel').forEach(function(td){set(td,false)}); refresh();
  });
  refresh();
})();
"""

_VOTE_CSS = """
body{font:14px/1.5 system-ui,sans-serif;margin:14px;background:#1e1e1e;color:#d4d4d4}
.legend{position:sticky;top:0;background:#1e1e1e;padding:8px 0;border-bottom:1px solid #444;z-index:3}
table{border-collapse:collapse;width:100%}
td,th{border:1px solid #3a3a3a;padding:.5rem;vertical-align:top;text-align:left}
img{max-width:340px;height:auto;image-rendering:pixelated;background:#fff}
pre{margin:0;white-space:pre-wrap;font:13px/1.5 ui-monospace,monospace}
mark{background:#5a2d2d;color:#ffd7d7;border-radius:2px}
.eng{cursor:pointer}
.eng:hover{outline:1px solid #666}
td.eq{opacity:.45}
td.wsdiff{background:#20262e}
td.misread{background:#2f2410}
.eng.sel{background:#1f3d1f;outline:2px solid #4caf50;opacity:1}
tr.ws td{background:#20262e}
tr.same{opacity:.4}
button{background:#333;color:#ddd;border:1px solid #555;border-radius:4px;padding:4px 10px;cursor:pointer}
"""


def _write_html(path: str, crops, ref_text, variants, src: str) -> None:
    """A crop-by-crop scoring page: the IMAGE the models were given, then every variant's
    text, each cell clickable to tally "this one read it right" (persisted in
    localStorage, like the compare harness's vote pages).

    This exists because char-sim can only say outputs DIFFER — deciding which is RIGHT
    needs the picture, since the reference is itself a model's guess. Rows are ordered
    by how much they deserve a human's attention: real disagreements first, then ones
    that differ only in whitespace/line breaks, then identical ones."""
    esc = html.escape
    labels = ["transformers"] + [l for l, _, _ in variants]

    rank = {"diff": 0, "ws": 1, "same": 2}
    graded = []
    for i, crop in enumerate(crops):
        texts = [ref_text[i] if ref_text else ""] + [v[1][i] for v in variants]
        if not ref_text or all(t == texts[0] for t in texts):
            cls = "same"
        elif all(_squash(t) == _squash(texts[0]) for t in texts):
            cls = "ws"      # same reading, different wrapping/spacing -> not a misread
        else:
            cls = "diff"
        graded.append((rank[cls], i, cls, crop, texts))
    graded.sort(key=lambda g: (g[0], g[1]))
    n = {c: sum(1 for g in graded if g[2] == c) for c in ("diff", "ws", "same")}

    P = ["<!doctype html><html lang='ja'><head><meta charset='utf-8'>",
         "<title>recognize compare</title>", f"<style>{_VOTE_CSS}</style></head><body>",
         "<div class='legend'><div id='tally'></div>",
         f"<div>{esc(src)} — {len(crops)} crops, cap {CAP_PIXELS}/{CAP_MODE} &nbsp;·&nbsp; "
         f"<b>{n['diff']}</b> 실제 불일치 &nbsp;·&nbsp; <b>{n['ws']}</b> 공백/줄바꿈만 다름 &nbsp;·&nbsp; "
         f"<b>{n['same']}</b> 동일 &nbsp;·&nbsp; <button id='reset'>선택 초기화</button></div>",
         "<div><small>맞게 읽은 칸을 클릭해 채점(다시 클릭하면 해제, 여러 칸 선택 가능). "
         "레퍼런스도 모델 출력이라 틀릴 수 있으니 그림을 보고 판단. "
         "칸 색: <span style='background:#2f2410;padding:0 4px'>오독(빨간 표시 = 다른 구간)</span> · "
         "<span style='background:#20262e;padding:0 4px'>공백/줄바꿈만 다름</span> · "
         "<span style='opacity:.45'>레퍼런스와 동일</span></small></div></div>",
         "<table><tr><th>#</th><th>crop (as sent)</th>"
         + "".join(f"<th>{esc(l)}</th>" for l in labels) + "</tr>"]
    for _r, i, cls, crop, texts in graded:
        P.append(f"<tr class='{cls}'><td>{i}</td>"
                 f"<td><img src='data:image/png;base64,{_png_b64(crop)}'><br>"
                 f"<small>{crop.width}×{crop.height} = {crop.width * crop.height // 1000}k px</small></td>")
        for k, (label, t) in enumerate(zip(labels, texts)):
            # Per-CELL grade, so a row flagged for one variant's misread still shows at a
            # glance which cells actually misread and which merely wrapped differently.
            if k == 0:
                cell, body = "", esc(t)
            elif t == texts[0]:
                cell, body = "eq", esc(t)
            elif _squash(t) == _squash(texts[0]):
                cell, body = "wsdiff", esc(t)   # same reading -> don't mark every space
            else:
                cell, body = "misread", _marked(texts[0], t, esc)
            P.append(f"<td class='eng {cell}' data-eng='{esc(label)}' data-key='{i}|{esc(label)}'>"
                     f"<pre>{body}</pre></td>")
        P.append("</tr>")
    P.append("</table>")
    P.append(f"<script>var engs={json.dumps(labels)};{_VOTE_JS}</script></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(P))


def _warn_cache(label: str, ms: list[float]) -> None:
    """Shout if any crop came back far under the median. Vision prefill dominates and is
    roughly flat per crop, so an outlier that fast means the server answered from its
    prompt cache — the exact artifact that faked a 8.6x and a 1.5x here. A backstop for
    the cache_prompt=False request flag, in case a server build ignores it."""
    if len(ms) < 4:
        return
    med, fast = statistics.median(ms), [x for x in ms if x < 0.25 * statistics.median(ms)]
    if fast:
        print(f"  !! [{label}] {len(fast)} crop(s) returned under 25% of the median "
              f"({min(fast):.0f}ms vs med {med:.0f}ms) — looks like prompt-cache hits, so this "
              f"rate is INFLATED. Restart llama-server and re-run.")


def _report_pass(label: str, ms: list[float], rows: list) -> float:
    rate = len(ms) / (sum(ms) / 1000)
    print(f"  [{label}] {rate:.3f} crops/sec | per-crop ms "
          f"min {min(ms):.0f} / med {statistics.median(ms):.0f} / max {max(ms):.0f}")
    rows.append(f"| {label} | {rate:.3f} | {min(ms):.0f} | {statistics.median(ms):.0f} | {max(ms):.0f} |")
    _warn_cache(label, ms)
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
    ap.add_argument("--rehtml", default="",
                    help="re-render a page this tool wrote (crops + texts are embedded in it) with the "
                         "current layout, without running any model or server. Needs --html for the output.")
    args = ap.parse_args()

    if args.rehtml:  # pure re-render: no crops to load, no endpoint to reach
        if not args.html:
            sys.exit("--rehtml needs --html <output path>")
        crops, ref_text, variants, src = _read_html(args.rehtml)
        _write_html(args.html, crops, ref_text, variants, src)
        print(f"re-rendered {len(crops)} crops from {args.rehtml} -> {args.html}")
        return 0

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
            _warn_cache(label, ms)
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
