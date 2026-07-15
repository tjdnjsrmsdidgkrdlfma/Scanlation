#!/usr/bin/env python3
"""Probe llama-server with the pipeline's EXACT translate request and classify a runaway.

The batch translate went runaway (~3300 tokens/request vs ~27 normal) on the schema
path with think off — and nobody has seen the runaway CONTENT yet: the pipeline only
surfaces the final result, and a request that never finishes leaves a plain client
with nothing at all. This probe makes the content observable two ways at once:
  (1) `max_tokens` turns "never ends" into "completed but truncated"
      (finish_reason=length), so the partial output comes back whole, and
  (2) `stream:true` prints tokens as they are generated, so even an aborted
      request leaves its transcript on the terminal.
Both bound the request (plus a wall deadline), so the probe can never create the
backlog/hang it is diagnosing.

It sends the same body the llama.cpp plugin sends (same prompt builders, same schema,
same chat_template_kwargs) straight to llama-server, then classifies the output:

  reasoning_content dominates  -> thinking despite enable_thinking:false — reasoning is
                                  generated OUTSIDE the grammar, the schema can't stop it
  content repeats a shingle    -> sampling loop — repetition inside a JSON string field
                                  is grammar-legal
  content isn't schema-shaped  -> response_format missing/ignored
  short + finish_reason=stop   -> terse: runaway NOT reproduced at this layer; compare
                                  the pipeline layer (replay its DEBUG-captured body)

    python tools/diag_runaway.py                      # pipeline shape: schema + think off
    python tools/diag_runaway.py --think              # flip enable_thinking
    python tools/diag_runaway.py --no-schema          # free generation (grammar A/B)
    python tools/diag_runaway.py --body captured.json # replay a DEBUG-captured plugin body

Run it where llama-server is reachable (the GPU host): --endpoint or LLAMACPP_ENDPOINT,
default http://127.0.0.1:8080 — this talks to llama-server directly, NOT the
scanlation server. Stdlib-only.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 - UTF-8 stdio (Japanese/Korean output on cp949 terminals)

import argparse
import json
import os
import sys
import time
import types
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

# prompt.py (and the context.py it pulls) are stdlib-only, but the package __init__
# is not — it re-exports the whole SDK, dragging numpy in. Register a stub package
# that skips __init__.py so the probe reuses the plugin's real prompt builders +
# schema while staying runnable on a bare GPU host.
_sdk_pkg = types.ModuleType("scanlation_sdk")
_sdk_pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "scanlation-sdk" / "scanlation_sdk")]
sys.modules.setdefault("scanlation_sdk", _sdk_pkg)
from scanlation_sdk.prompt import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    batch_schema,
    build_batch_prompt,
)

# A bubble-sized sample page: enough texts to exercise the batch schema (t0..t3).
SAMPLE_TEXTS = ["おはよう!", "何してるの?", "それは秘密です", "行くぞ!!"]

# Classification thresholds. Normal schema+think-off output is ~27 completion tokens
# and a runaway is thousands, so 100 separates them with a wide margin either way.
TERSE_TOKENS = 100
# A 24-char shingle repeating 5+ times in the 2000-char tail is a loop, not prose.
LOOP_SHINGLE, LOOP_TAIL, LOOP_MIN = 24, 2000, 5


def _pipeline_body(args: argparse.Namespace) -> dict:
    """The llama.cpp plugin's request body, same shape as plugin._body() +
    _translate_batch_call(). Sampling values match COMMON_LLM_OPTIONS defaults."""
    texts = args.texts or SAMPLE_TEXTS
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": build_batch_prompt(texts, args.src, args.dst)},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "stream": False,  # overridden below; kept so a --dump of the body mirrors the plugin
        "chat_template_kwargs": {"enable_thinking": args.think},
    }
    if not args.no_schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "translations", "schema": batch_schema(len(texts)), "strict": True},
        }
    return body


def _stream(url: str, body: dict, deadline: float) -> tuple[dict, str | None, dict | None, dict | None, str]:
    """POST and consume the SSE stream, printing deltas live. Returns
    (parts, finish_reason, usage, timings, aborted_reason)."""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=deadline)
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} from {url}: {e.read().decode('utf-8', 'replace')[:500]}")
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"cannot reach {url}: {e}")

    parts = {"reasoning_content": "", "content": ""}
    finish = usage = timings = None
    aborted, channel = "", None
    t_end = time.monotonic() + deadline
    try:
        with resp:
            for raw in resp:
                if time.monotonic() > t_end:
                    aborted = f"wall deadline {deadline:.0f}s"
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("error"):
                    aborted = f"server error: {chunk['error']}"
                    break
                usage = chunk.get("usage") or usage
                timings = chunk.get("timings") or timings
                for ch in chunk.get("choices") or []:
                    finish = ch.get("finish_reason") or finish
                    delta = ch.get("delta") or {}
                    for key in ("reasoning_content", "content"):
                        piece = delta.get(key)
                        if not piece:
                            continue
                        if channel != key:  # mark channel switches so thinking is visibly separate
                            channel = key
                            print(f"\n──── {key} ────")
                        parts[key] += piece
                        sys.stdout.write(piece)
                        sys.stdout.flush()
    except (TimeoutError, urllib.error.URLError, OSError) as e:
        aborted = f"stream cut: {e}"  # whatever arrived is already printed AND in parts
    print()
    return parts, finish, usage, timings, aborted


def _loop_signature(text: str) -> tuple[str, int]:
    """Most-repeated shingle in the tail — a high count is the tell of a sampling
    loop (the grammar can't prevent repetition inside a string field)."""
    tail = text[-LOOP_TAIL:]
    if len(tail) <= LOOP_SHINGLE:
        return "", 0
    counts = Counter(tail[i:i + LOOP_SHINGLE] for i in range(len(tail) - LOOP_SHINGLE))
    return counts.most_common(1)[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("texts", nargs="*",
                    help=f"texts to translate (default: {len(SAMPLE_TEXTS)} sample bubbles)")
    ap.add_argument("--src", default="ja", help="source language code (default ja)")
    ap.add_argument("--dst", default="ko", help="target language code (default ko)")
    ap.add_argument("--model", default="", help="model id (llama-server ignores it)")
    ap.add_argument("--think", action="store_true",
                    help="send enable_thinking:true (default false = the pipeline default)")
    ap.add_argument("--no-schema", action="store_true",
                    help="drop response_format (schema-vs-free A/B)")
    ap.add_argument("--body", help="replay a captured request-body JSON file instead of composing one")
    ap.add_argument("--endpoint", default=os.getenv("LLAMACPP_ENDPOINT", "http://127.0.0.1:8080"),
                    help="llama-server base URL (default $LLAMACPP_ENDPOINT or http://127.0.0.1:8080)")
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="server-side cap: bounds the probe AND makes a runaway come back whole")
    ap.add_argument("--deadline", type=float, default=120.0,
                    help="wall/read seconds before the stream is abandoned")
    ap.add_argument("--dump", help="write the body + collected output as JSON here")
    args = ap.parse_args()

    if args.body:
        body = json.loads(Path(args.body).read_text(encoding="utf-8"))
        if args.texts or args.think or args.no_schema:
            print("(--body replay: texts/--think/--no-schema ignored — the captured body wins)")
    else:
        body = _pipeline_body(args)
    # Bounds — the probe must never become the backlog it is diagnosing.
    prior_cap = body.get("max_tokens")
    body["max_tokens"] = min(prior_cap, args.max_tokens) if isinstance(prior_cap, int) else args.max_tokens
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}

    endpoint = args.endpoint.rstrip("/")
    schema_used = "response_format" in body
    think = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
    print(f"POST {endpoint}/v1/chat/completions  "
          f"[{'schema' if schema_used else 'free'}, enable_thinking={think}, "
          f"max_tokens={body['max_tokens']}, deadline={args.deadline:.0f}s]")

    t0 = time.monotonic()
    parts, finish, usage, timings, aborted = _stream(
        f"{endpoint}/v1/chat/completions", body, args.deadline)
    wall = time.monotonic() - t0

    r, c = parts["reasoning_content"], parts["content"]
    comp = (usage or {}).get("completion_tokens")
    sig, sig_n = _loop_signature(r + c)
    json_ok = None
    if schema_used and c:
        try:
            json.loads(c)
            json_ok = True
        except ValueError:
            json_ok = False

    print("\n== 분석 ==")
    line = f"  finish={finish or '?'}  wall {wall:.1f}s"
    if aborted:
        line += f"  (중단: {aborted})"
    if usage:
        line += f"  completion_tokens={comp} (prompt {usage.get('prompt_tokens')})"
    print(line)
    if timings and timings.get("predicted_per_second"):
        print(f"  decode {timings['predicted_per_second']:.1f} t/s")
    print(f"  reasoning {len(r)}자 / content {len(c)}자")
    if sig_n >= LOOP_MIN:
        print(f"  반복 시그니처: {sig!r} x{sig_n} (tail {LOOP_TAIL}자 기준)")
    if json_ok is not None:
        note = "" if json_ok else (" — finish=length 잘림이면 그 자체는 정상" if finish == "length" else "")
        print(f"  content JSON: {'유효' if json_ok else '불완전' + note}")

    # Token count when the server didn't send usage: ko/ja run roughly 2 chars/token.
    tokens = comp if comp is not None else (len(r) + len(c)) // 2
    if finish == "stop" and tokens < TERSE_TOKENS:
        verdict = ("정상(terse) — 이 층(모델 직타)에선 폭주 재현 안 됨. 파이프라인 층을 의심: "
                   "DEBUG로 캡처한 plugin body를 --body로 재생해 비교.")
    elif len(r) >= len(c):
        verdict = ("생각(thinking) — reasoning은 grammar 밖이라 schema로 못 막음. "
                   "enable_thinking 전달 경로(신버전 플러그인 설치 여부)를 점검하고, "
                   "모델이 무시하면 --reasoning-budget 0 (translate-gpu-mi50.md §진단 방법론 2b).")
    elif sig_n >= LOOP_MIN:
        verdict = ("반복(loop) — 문자열 필드 안 반복은 grammar 합법. "
                   "repeat/presence penalty(또는 DRY) 조절 (translate-gpu-mi50.md §진단 방법론 2c).")
    elif schema_used and not c.lstrip().startswith("{"):
        verdict = "schema 미적용 — content가 스키마 JSON 형태가 아님. response_format 전송/서버 지원 확인."
    else:
        verdict = "본문 장황 — 생각·반복 아님, JSON 형태인데 김. 스키마·프롬프트 내용 점검."
    print(f"  → 판정: {verdict}")

    if args.dump:
        Path(args.dump).write_text(json.dumps({
            "endpoint": endpoint, "body": body, "finish_reason": finish, "aborted": aborted,
            "wall_s": round(wall, 1), "usage": usage, "timings": timings,
            "reasoning_content": r, "content": c,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  덤프: {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
