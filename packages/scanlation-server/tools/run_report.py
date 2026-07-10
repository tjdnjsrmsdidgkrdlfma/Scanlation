#!/usr/bin/env python3
"""Drive the running server's /run_pipeline/ on local image(s) like the extension,
then export a REPORT — per-region OCR (source) + final translation (destination),
per-stage timing, and the active engine/option config — as Markdown + a JSON sidecar.

Where run_image.py just prints a one-line-per-region summary for log diagnosis, this
SAVES a structured report you can eyeball or diff across a change (before/after a
resolution cap, an engine swap, a prompt edit...). It reads the per-stage `timing`
the server returns, so decode / detect+recognize / translate are broken out, and the
aggregate puts detect+recognize next to translate — i.e. whether the run is
translate-bound.

Runs SERIAL by default so the per-stage timings are clean (concurrent images contend
on the GPU lock + translate semaphore, which pollutes lockwait/semwait). Pass
--parallel to instead fire every image at once, reproducing how the extension hits a
page that loads many images (concurrent-load behaviour); the report notes the mode.

Visual overlays are intentionally out of scope — detection is fixed and rendered
results already come back as images.

  # every image under a folder, forced fresh, clean serial timing (default)
  python tools/run_report.py samples/

  # reproduce concurrent page load; custom output prefix
  python tools/run_report.py samples/ --parallel --out mybench

  # specific files, remote server behind nginx + token
  python tools/run_report.py a.png b.png --server https://scan.example.com --token SECRET

  # verify the report formatting without a server (canned data)
  python tools/run_report.py --selftest

Server URL: --server or SCANLATION_SERVER (default http://127.0.0.1:4010). Auth token:
--token or SCANLATION_AUTH_TOKEN. Only the Python 3 stdlib is used, so it runs
anywhere without installing anything.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Reuse the image-expansion walk from run_image.py (same tools/ dir, stdlib-only).
from run_image import iter_images

# A bare urllib UA trips Cloudflare's Browser Integrity Check; a real browser UA
# passes it (same reasoning as run_image.py). Hitting the origin directly sidesteps CF.
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# The per-stage timing keys the server returns (orchestrator.run_page), in report order.
# detect_ms + recognize_ms are the two halves inside detect_recognize_ms (the whole GPU
# span — it also covers engine resolve/first-load, so it stays >= detect + recognize).
_STAGES = ["decode_ms", "lockwait_ms", "detect_recognize_ms", "detect_ms", "recognize_ms",
           "semwait_ms", "translate_ms", "total_ms"]


# --- HTTP (stdlib) --------------------------------------------------------------
def _headers(token: str, *, json_body: bool) -> dict:
    h = {"User-Agent": _UA}
    if json_body:
        h["Content-Type"] = "application/json"
    if token:
        h["X-Auth-Token"] = token
    return h


def post_pipeline(server: str, token: str, img: Path, force: bool) -> dict:
    """POST one image to /run_pipeline/ and return the FULL parsed response
    (``{result, timing?}``). md5 is over the base64 string, matching the extension."""
    b64 = base64.b64encode(img.read_bytes()).decode("ascii")
    md5 = hashlib.md5(b64.encode("utf-8")).hexdigest()
    body = json.dumps({"md5": md5, "contents": b64, "force": force}).encode("utf-8")
    req = urllib.request.Request(f"{server}/run_pipeline/", data=body,
                                 headers=_headers(token, json_body=True))
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get_settings(server: str, token: str) -> dict | None:
    """GET /get_settings/ for the report header; None if it can't be fetched
    (older server / auth / offline) so the report degrades instead of failing."""
    req = urllib.request.Request(f"{server}/get_settings/", headers=_headers(token, json_body=False))
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except Exception:  # noqa: BLE001 - header is best-effort
        return None


# --- pure summarisation / report building (server-free, unit-testable) ----------
def summarize_settings(settings: dict | None) -> dict | None:
    """Compact 'which engines + options produced this' header from /get_settings/.
    Records each active engine's device + persisted option overrides (what
    distinguishes one A/B config from another) — not the schema-merged effective set."""
    if not settings:
        return None
    sel = settings.get("selection", {})
    engines = settings.get("engines", {})
    out = {
        "languages": f"{sel.get('lang_src')}->{sel.get('lang_dst')}",
        "prompt_active": sel.get("prompt_active"),
        "engines": {},
    }
    for role in ("detector", "recognizer", "translator"):
        name = sel.get(role, "")
        entry = next((e for e in engines.get(role, []) if e.get("name") == name), None) or {}
        out["engines"][role] = {
            "name": name,
            "device": entry.get("device"),
            "options": entry.get("options", {}),
        }
    return out


def _mean(vals: list[float]) -> float | None:
    return round(sum(vals) / len(vals), 1) if vals else None


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    mid = len(s) // 2
    m = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
    return round(m, 1)


def build_report(runs: list[dict], settings_summary: dict | None, meta: dict) -> dict:
    """Assemble the canonical report object (the JSON sidecar). ``runs`` is a list of
    {image, ok, regions, timing, error}. The Markdown renders from this same object."""
    ok = [r for r in runs if r["ok"]]
    failed = [r for r in runs if not r["ok"]]
    timed = [r for r in ok if r.get("timing")]
    # Per-stage sample lists (one value per timed image); every aggregate stat reads off these.
    vals = {k: [r["timing"].get(k, 0) for r in timed] for k in _STAGES}
    timing_sum = {k: round(sum(v), 1) for k, v in vals.items()} if timed else {}
    timing_mean = {k: _mean(v) for k, v in vals.items()} if timed else {}
    timing_median = {k: _median(v) for k, v in vals.items()} if timed else {}
    timing_min = {k: round(min(v), 1) for k, v in vals.items()} if timed else {}
    timing_max = {k: round(max(v), 1) for k, v in vals.items()} if timed else {}
    return {
        "generated_at": meta["generated_at"],
        "server": meta["server"],
        "mode": meta["mode"],          # "serial" | "parallel" (timing is contended when parallel)
        "settings": settings_summary,
        "images": runs,
        "aggregate": {
            "images": len(runs),
            "ok": len(ok),
            "failed": len(failed),
            "regions_total": sum(len(r.get("regions") or []) for r in ok),
            "timing_sum": timing_sum,
            "timing_mean": timing_mean,
            "timing_median": timing_median,
            "timing_min": timing_min,
            "timing_max": timing_max,
        },
    }


def _md_cell(text: str) -> str:
    """Make a string safe for one Markdown table cell: escape pipes, flatten newlines."""
    return str(text).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ⏎ ")


def _fmt_timing(t: dict | None) -> str:
    if not t:
        return "(no timing — cache hit? use a fresh run)"
    return (f"total {t.get('total_ms')}ms  (decode {t.get('decode_ms')} / "
            f"detect+recognize {t.get('detect_recognize_ms')} "
            f"[detect {t.get('detect_ms')} + recognize {t.get('recognize_ms')}] / "
            f"translate {t.get('translate_ms')}"
            f"; lockwait {t.get('lockwait_ms')}, semwait {t.get('semwait_ms')})")


def build_markdown(report: dict) -> str:
    L: list[str] = []
    agg = report["aggregate"]
    L.append(f"# 번역 실행 리포트 — {report['generated_at']}")
    L.append("")
    L.append(f"- 서버: `{report['server']}`  ·  모드: **{report['mode']}**"
             + ("" if report["mode"] == "serial"
                else "  ⚠ 병렬 실행이라 lockwait/semwait 등 단계별 시간은 이미지 간 경합으로 오염됨"))
    L.append(f"- 이미지: {agg['images']} (성공 {agg['ok']}, 실패 {agg['failed']})  ·  총 리전 {agg['regions_total']}")

    s = report.get("settings")
    if s:
        L.append("")
        L.append("## 설정")
        L.append(f"- 언어: `{s['languages']}`  ·  프롬프트: `{s['prompt_active']}`")
        for role in ("detector", "recognizer", "translator"):
            e = s["engines"][role]
            opts = ", ".join(f"{k}={v}" for k, v in (e.get("options") or {}).items()) or "(기본)"
            dev = e.get("device") or "기본"
            L.append(f"- {role}: **{e['name'] or '(미선택)'}**  ·  device `{dev}`  ·  옵션: {opts}")
    else:
        L.append("")
        L.append("## 설정\n(`/get_settings/`를 못 읽어 헤더 생략)")

    mean = agg.get("timing_mean") or {}
    if mean:
        med, lo, hi = agg.get("timing_median") or {}, agg.get("timing_min") or {}, agg.get("timing_max") or {}
        L.append("")
        L.append("## 집계 — 단계별 시간 (성공 실행 기준)")
        L.append("")
        L.append("| stage | 합 ms | 평균 ms | 중앙값 ms | 최소 ms | 최대 ms |")
        L.append("|---|---|---|---|---|---|")
        for k in _STAGES:
            L.append(f"| {k} | {agg['timing_sum'].get(k)} | {mean.get(k)} "
                     f"| {med.get(k)} | {lo.get(k)} | {hi.get(k)} |")
        L.append("")
        L.append("_detect_recognize_ms는 GPU 반쪽 전체(엔진 resolve·first-load 포함)라 "
                 "detect_ms+recognize_ms보다 큼. recognize_ms가 엔진 교체·최적화 대상._")
        dr, tr = mean.get("detect_recognize_ms"), mean.get("translate_ms")
        if dr is not None and tr is not None:
            verdict = "translate-bound (번역이 지배)" if tr > dr else "detect+recognize-bound (인식이 지배)"
            L.append("")
            L.append(f"→ 평균 detect+recognize **{dr}ms** vs translate **{tr}ms** ⇒ **{verdict}**"
                     + ("" if report["mode"] == "serial" else "  (병렬 모드라 참고용)"))

        # Per-case timing (image = one translate batch = finest granularity available).
        # recognize is surfaced next to translate because it's the stage being swapped/tuned.
        cases = [(r["image"], r["timing"]) for r in report["images"] if r["ok"] and r.get("timing")]
        if cases:
            L.append("")
            L.append("### 케이스별 시간 (이미지 = 배치 1건)")
            L.append("")
            L.append("| 이미지 | recognize ms | translate ms | total ms | regions |")
            L.append("|---|---|---|---|---|")
            for name, t in cases:
                L.append(f"| {_md_cell(name)} | {t.get('recognize_ms')} | {t.get('translate_ms')} "
                         f"| {t.get('total_ms')} | {t.get('regions')} |")

    L.append("")
    L.append("## 이미지별")
    for r in report["images"]:
        L.append("")
        if not r["ok"]:
            L.append(f"### {r['image']} — 실패")
            L.append(f"- {r.get('error')}")
            continue
        regions = r.get("regions") or []
        L.append(f"### {r['image']} — {len(regions)} regions")
        L.append(f"- {_fmt_timing(r.get('timing'))}")
        L.append("")
        L.append("| # | bounds | source (OCR) | destination (번역) |")
        L.append("|---|---|---|---|")
        for i, it in enumerate(regions):
            L.append(f"| {i} | {it.get('bounds')} | {_md_cell(it.get('source', ''))} "
                     f"| {_md_cell(it.get('destination', ''))} |")

    if agg["failed"]:
        L.append("")
        L.append("## 실패")
        for r in report["images"]:
            if not r["ok"]:
                L.append(f"- `{r['image']}`: {r.get('error')}")

    L.append("")
    return "\n".join(L)


# --- driver ---------------------------------------------------------------------
def _run_one(server: str, token: str, img: Path, force: bool) -> dict:
    """Drive one image; never raises — failures are captured into the run record."""
    try:
        resp = post_pipeline(server, token, img, force)
        return {"image": img.name, "ok": True,
                "regions": resp.get("result") or [], "timing": resp.get("timing"), "error": None}
    except urllib.error.HTTPError as e:
        return {"image": img.name, "ok": False, "regions": [], "timing": None,
                "error": f"HTTP {e.code} {e.read().decode('utf-8', 'replace')[:200]}"}
    except Exception as e:  # noqa: BLE001 - network/decoding; capture and move on
        return {"image": img.name, "ok": False, "regions": [], "timing": None, "error": f"ERROR {e}"}


def _selftest() -> int:
    """Render canned data through build_report/build_markdown so the formatting can be
    checked without a running server (server-free verification of Part 2)."""
    runs = [
        {"image": "p1.png", "ok": True,
         "regions": [{"bounds": [1, 2, 3, 4], "source": "こんにちは", "destination": "안녕하세요"},
                     {"bounds": [5, 6, 7, 8], "source": "a|b\nc", "destination": "표 | 안전"}],
         "timing": {"decode_ms": 5.0, "lockwait_ms": 0.1, "detect_recognize_ms": 40.0,
                    "detect_ms": 12.0, "recognize_ms": 26.0,  # sum 38 <= 40 (umbrella)
                    "semwait_ms": 0.0, "translate_ms": 900.0, "total_ms": 945.1, "regions": 2},
         "error": None},
        {"image": "bad.png", "ok": False, "regions": [], "timing": None, "error": "HTTP 400 bad image"},
    ]
    settings = summarize_settings({
        "selection": {"lang_src": "ja", "lang_dst": "ko", "prompt_active": "default",
                      "detector": "comic-text-and-bubble-detector", "recognizer": "manga-ocr",
                      "translator": "Ollama"},
        "engines": {"recognizer": [{"name": "manga-ocr", "device": "cpu", "options": {}}],
                    "translator": [{"name": "Ollama", "device": None, "options": {"model": "gemma"}}]},
    })
    report = build_report(runs, settings, {"generated_at": "SELFTEST", "server": "http://x", "mode": "serial"})
    md = build_markdown(report)
    # invariants
    agg = report["aggregate"]
    stage_vals = {"decode_ms": 5.0, "lockwait_ms": 0.1, "detect_recognize_ms": 40.0,
                  "detect_ms": 12.0, "recognize_ms": 26.0,
                  "semwait_ms": 0.0, "translate_ms": 900.0, "total_ms": 945.1}
    assert (agg["images"], agg["ok"], agg["failed"], agg["regions_total"]) == (2, 1, 1, 2), agg
    # one timed run → sum/mean/median/min/max all collapse to that single sample
    for key in ("timing_sum", "timing_mean", "timing_median", "timing_min", "timing_max"):
        assert agg[key] == stage_vals, (key, agg[key])
    assert "안녕하세요" in md and "こんにちは" in md          # OCR + translation rendered
    assert "translate-bound" in md                          # 900 > 40 verdict
    assert "중앙값" in md and "최소" in md and "최대" in md    # per-stage spread columns
    assert "recognize_ms" in md and "detect 12.0 + recognize 26.0" in md  # detect/recognize split
    assert "케이스별 시간" in md                              # per-case timing list
    assert "표 \\| 안전" in md and "a\\|b ⏎ c" in md          # pipe/newline escaping in cells
    assert "manga-ocr" in md and "model=gemma" in md         # settings header
    assert json.dumps(report, ensure_ascii=False)           # JSON-serialisable
    print("selftest OK - build_report/build_markdown produce a well-formed report")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a /run_pipeline/ report (OCR + translation + per-stage timing).")
    ap.add_argument("images", nargs="*", help="image files and/or folders")
    ap.add_argument("--server", default=os.environ.get("SCANLATION_SERVER", "http://127.0.0.1:4010"),
                    help="server base URL (default: %(default)s)")
    ap.add_argument("--token", default=os.environ.get("SCANLATION_AUTH_TOKEN", ""),
                    help="X-Auth-Token if the server requires one")
    ap.add_argument("--parallel", action="store_true",
                    help="fire all images at once (reproduces concurrent page load; pollutes per-stage timing)")
    ap.add_argument("--use-cache", action="store_true",
                    help="honor the cache (default: force a fresh run so timing is present)")
    ap.add_argument("--out", default=None, help="output prefix (default: run_report_<timestamp>)")
    ap.add_argument("--selftest", action="store_true", help="verify report formatting on canned data, no server")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if not a.images:
        print("no images given (pass files/folders, or --selftest)", file=sys.stderr)
        return 2

    server = a.server.rstrip("/")
    force = not a.use_cache
    mode = "parallel" if a.parallel else "serial"
    imgs = list(iter_images(a.images))
    if not imgs:
        print("no images found", file=sys.stderr)
        return 1
    print(f"POST {server}/run_pipeline/  ({len(imgs)} images, {mode}, force={force})", file=sys.stderr)

    settings_summary = summarize_settings(get_settings(server, a.token))

    if a.parallel:
        # Key futures by input index, not img.name: rglob can yield the same basename
        # from different folders, and a name-keyed dict would collide (one result lost,
        # another duplicated) while the count still looked right.
        with ThreadPoolExecutor(max_workers=len(imgs)) as ex:
            futs = {ex.submit(_run_one, server, a.token, img, force): i for i, img in enumerate(imgs)}
            done = {futs[f]: f.result() for f in as_completed(futs)}
        runs = [done[i] for i in range(len(imgs))]  # restore input order
    else:
        runs = [_run_one(server, a.token, img, force) for img in imgs]

    meta = {"generated_at": datetime.now().isoformat(timespec="seconds"), "server": server, "mode": mode}
    report = build_report(runs, settings_summary, meta)

    prefix = a.out or f"run_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    md_path, json_path = Path(f"{prefix}.md"), Path(f"{prefix}.json")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    agg = report["aggregate"]
    mean = agg.get("timing_mean") or {}
    print(f"wrote {md_path}  and  {json_path}", file=sys.stderr)
    print(f"  images ok={agg['ok']}/{agg['images']}  regions={agg['regions_total']}", file=sys.stderr)
    if mean.get("detect_recognize_ms") is not None:
        print(f"  mean detect+recognize={mean['detect_recognize_ms']}ms  translate={mean['translate_ms']}ms",
              file=sys.stderr)
    return 1 if agg["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
