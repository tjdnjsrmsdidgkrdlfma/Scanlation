#!/usr/bin/env python3
"""Sustained-load thermal probe — does junction plateau, or keep climbing?

A single chapter is known safe: 21 pages at P4 peaks at junction 74-78C and is
over in ~13-16s. The open question is CONTINUOUS use — the extension fires a
whole page's images at once, so back-to-back chapters never let the card idle.
This runs N chapters with no gap, samples the MI50's sensors twice a second,
and reports whether the per-chapter peak FLATTENS (equilibrium) or steps up
every chapter.

Reuses bench_translate_concurrency for the request bodies, so the load is the
same production-shaped batch the plugin sends (same prompts, schema, DRY).

HARD CEILING: junction must never reach 95C. The guards below stop the load
generator far short of that, because a cut is never instant — detection lag
plus the drain of in-flight requests keeps the card heating for a couple of
seconds after the decision. Budget at the measured ~0.94 C/s (MoE at 150W):

    sustained  junction >= 85C on 2 consecutive samples
    spike      junction >= 88C on a single sample
    predictive slope over the last 3s, projected 3s ahead, >= 91C
    memory     mem >= 80C on 2 consecutive samples (HBM crit is 94)

Worst case is roughly 88 + 3s * 1 C/s = 91C, leaving 4C of headroom. Tripping
is not a failed run: above 85 the answer to the question is already "no safe
plateau", so there is nothing left to learn by climbing further.

A trip STOPS SUBMITTING and lets in-flight requests drain. It never signals
llama-server — killing a process mid-GPU-work wedges the amdgpu context and
takes a cold boot to clear (see translate-gpu-mi50.md section 4).

Fans are read-only here. This tool never writes pwm, and never touches the
power cap; whatever fancontrol is doing stays untouched.

    python tools/bench_translate_sustained.py run_report_20260716_004042.md
    python tools/bench_translate_sustained.py report.md --chapters 8 -P 2
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 - UTF-8 stdio

import argparse
import glob
import json
import os
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bench_translate_concurrency as btc

MI50_PCI = "0000:03:00.0"


def _read_int(path: str) -> int | None:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


class Sensors:
    """MI50 hwmon by PCI path (hwmon index moves every boot) + nct6687 fan, read-only."""

    def __init__(self, pci: str = MI50_PCI):
        hits = glob.glob(f"/sys/bus/pci/devices/{pci}/hwmon/hwmon*/")
        if not hits:
            sys.exit(f"no hwmon under /sys/bus/pci/devices/{pci} — wrong PCI address?")
        self.gpu = hits[0]
        self.fan = None
        for h in glob.glob("/sys/class/hwmon/hwmon*/"):
            try:
                with open(h + "name") as f:
                    if f.read().strip() == "nct6687":
                        self.fan = h
                        break
            except OSError:
                pass

    def read(self) -> dict:
        g = self.gpu
        s = {
            "edge": _read_int(g + "temp1_input"),
            "junction": _read_int(g + "temp2_input"),
            "mem": _read_int(g + "temp3_input"),
            "power": _read_int(g + "power1_input"),
        }
        for k in ("edge", "junction", "mem"):
            if s[k] is not None:
                s[k] /= 1000.0
        if s["power"] is not None:
            s["power"] /= 1_000_000.0
        s["fan_rpm"] = _read_int(self.fan + "fan4_input") if self.fan else None
        s["fan_pwm"] = _read_int(self.fan + "pwm4") if self.fan else None
        return s

    def limits(self) -> dict:
        g = self.gpu
        return {
            "junction_crit": (_read_int(g + "temp2_crit") or 0) / 1000.0,
            "mem_crit": (_read_int(g + "temp3_crit") or 0) / 1000.0,
            "power_cap": (_read_int(g + "power1_cap") or 0) / 1_000_000.0,
        }


class Guard(threading.Thread):
    """Samples sensors; sets `tripped` when any ceiling rule fires."""

    def __init__(self, sensors: Sensors, args, log_path: Path):
        super().__init__(daemon=True)
        self.s = sensors
        self.a = args
        self.log_path = log_path
        self.stop_flag = threading.Event()   # set by main when the run is done
        self.tripped = threading.Event()     # set by us when a ceiling fires
        self.reason = ""
        self.samples: list[tuple[float, dict]] = []
        self.t0 = time.monotonic()

    def _projected(self, now: float) -> float | None:
        """Least-squares projection `horizon` seconds ahead over the recent window.

        Only meaningful once the card has reached its load temperature. Going from
        idle to load is a step — junction covers 53->73C in half a second — and
        extrapolating that transient predicts 130C every single run. The arm
        threshold keeps this guard in the range it was written for: the slow climb
        between the load plateau and the ceiling.
        """
        win = [(t, d["junction"]) for t, d in self.samples[-60:]
               if now - t <= self.a.slope_window and d["junction"] is not None]
        if len(win) < 5 or win[-1][1] < self.a.project_arm:
            return None
        dt = win[-1][0] - win[0][0]
        if dt <= 1.0:
            return None
        n = len(win)
        mt = sum(t for t, _ in win) / n
        mj = sum(j for _, j in win) / n
        denom = sum((t - mt) ** 2 for t, _ in win)
        if denom <= 0:
            return None
        slope = sum((t - mt) * (j - mj) for t, j in win) / denom
        return win[-1][1] + slope * self.a.horizon

    def run(self) -> None:
        hot_j = hot_m = 0
        with open(self.log_path, "w", encoding="utf-8") as log:
            log.write("t,junction,mem,edge,power_w,fan_rpm,fan_pwm\n")
            while not self.stop_flag.is_set():
                now = time.monotonic()
                d = self.s.read()
                self.samples.append((now, d))
                log.write("{:.2f},{},{},{},{},{},{}\n".format(
                    now - self.t0, d["junction"], d["mem"], d["edge"],
                    d["power"], d["fan_rpm"], d["fan_pwm"]))
                log.flush()

                # Keep sampling after a trip: the drain of in-flight requests is
                # exactly the window where the ceiling could still be breached, so
                # it has to end up in the log too.
                if not self.tripped.is_set():
                    j, m = d["junction"], d["mem"]
                    if j is None:
                        # A sensor that stops reading is not a reason to keep pushing.
                        self._trip("junction unreadable")
                    else:
                        hot_j = hot_j + 1 if j >= self.a.cut else 0
                        hot_m = hot_m + 1 if (m is not None and m >= self.a.cut_mem) else 0
                        p = self._projected(now)
                        if j >= self.a.cut_hard:
                            self._trip(f"junction {j:.0f}C >= hard {self.a.cut_hard}C")
                        elif hot_j >= 2:
                            self._trip(f"junction {j:.0f}C >= {self.a.cut}C twice running")
                        elif hot_m >= 2:
                            self._trip(f"mem {m:.0f}C >= {self.a.cut_mem}C twice running")
                        elif p is not None and p >= self.a.cut_project:
                            self._trip(f"junction {j:.0f}C trending to {p:.0f}C "
                                       f"in {self.a.horizon:.0f}s >= {self.a.cut_project}C")
                time.sleep(self.a.interval)

    def _trip(self, reason: str) -> None:
        self.reason = reason
        self.tripped.set()
        print(f"\n!! THERMAL CUT — {reason}. No new requests; draining in-flight.",
              flush=True)

    def window(self, t_from: float, t_to: float) -> list[dict]:
        return [d for t, d in self.samples if t_from <= t <= t_to]


def _peak(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r[key] is not None]
    return max(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", help="run_report markdown with per-page source tables")
    ap.add_argument("--chapters", type=int, default=5, help="back-to-back passes (default 5)")
    ap.add_argument("-P", "--concurrency", type=int, default=4,
                    help="in-flight requests, matching the production gate (default 4)")
    ap.add_argument("--gap", type=float, default=0.0,
                    help="idle seconds between chapters (default 0 = worst case)")
    ap.add_argument("--start-max", type=float, default=70.0,
                    help="refuse to start above this junction (default 70)")
    ap.add_argument("--cut", type=float, default=85.0, help="sustained cut, 2 samples (default 85)")
    ap.add_argument("--cut-hard", type=float, default=88.0, help="single-sample cut (default 88)")
    ap.add_argument("--cut-project", type=float, default=91.0,
                    help="projected-temperature cut (default 91)")
    ap.add_argument("--cut-mem", type=float, default=80.0, help="mem cut, 2 samples (default 80)")
    ap.add_argument("--interval", type=float, default=0.5, help="sample seconds (default 0.5)")
    ap.add_argument("--slope-window", type=float, default=5.0, help="slope window s (default 5)")
    ap.add_argument("--horizon", type=float, default=3.0, help="projection horizon s (default 3)")
    ap.add_argument("--project-arm", type=float, default=78.0,
                    help="only project above this junction, past the idle->load step (default 78)")
    ap.add_argument("--src", default="ja")
    ap.add_argument("--dst", default="ko")
    ap.add_argument("--model", default="")
    ap.add_argument("--endpoint", default=os.getenv("LLAMACPP_ENDPOINT", "http://127.0.0.1:8080"))
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--log", default="sustained_thermal.csv")
    args = ap.parse_args()

    if args.cut_hard >= 95 or args.cut >= 95 or args.cut_project >= 95:
        sys.exit("refusing to run: every cut must sit below the 95C ceiling")

    pages = btc.parse_pages(Path(args.report))
    if not pages:
        sys.exit(f"no page tables found in {args.report}")
    body_args = types.SimpleNamespace(
        model=args.model, src=args.src, dst=args.dst, max_tokens=args.max_tokens)
    bodies = [btc._body(texts, body_args) for _, texts in pages]
    url = f"{args.endpoint.rstrip('/')}/v1/chat/completions"

    sensors = Sensors()
    lim = sensors.limits()
    now = sensors.read()
    print(f"MI50 hwmon {sensors.gpu}")
    print(f"  limits: junction crit {lim['junction_crit']:.0f}C · mem crit {lim['mem_crit']:.0f}C "
          f"· power cap {lim['power_cap']:.0f}W")
    print(f"  cuts:   sustained {args.cut}C · spike {args.cut_hard}C · projected "
          f"{args.cut_project}C (armed above {args.project_arm}C) · mem {args.cut_mem}C "
          f" (ceiling 95C)")
    print(f"  start:  junction {now['junction']:.0f}C · mem {now['mem']:.0f}C · "
          f"fan {now['fan_rpm']} rpm (read-only)")
    if now["junction"] is None or now["junction"] > args.start_max:
        sys.exit(f"junction {now['junction']}C is above --start-max {args.start_max}C — "
                 f"let the card settle so chapter 1 is comparable")
    print(f"{len(bodies)} pages/chapter · {args.chapters} chapters · P={args.concurrency} "
          f"· gap {args.gap}s\n")

    guard = Guard(sensors, args, Path(args.log))
    guard.start()

    def post(body):
        if guard.tripped.is_set():
            return None
        return btc._post(url, body, args.timeout)

    rows = []
    try:
        for ch in range(1, args.chapters + 1):
            if guard.tripped.is_set():
                break
            t_start = time.monotonic()
            j_start = sensors.read()["junction"]
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                results = list(ex.map(post, bodies))
            t_end = time.monotonic()
            done = [r for r in results if r]
            toks = sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in done)
            w = guard.window(t_start, t_end)
            row = {
                "ch": ch,
                "wall": t_end - t_start,
                "ok": len(done),
                "toks": toks,
                "j_start": j_start,
                "j_peak": _peak(w, "junction"),
                "m_peak": _peak(w, "mem"),
                "p_peak": _peak(w, "power"),
                "fan": _peak(w, "fan_rpm"),
            }
            rows.append(row)
            print("ch{ch}: {wall:5.1f}s  {ok:2d}/{n} pages  {toks:4d} tok  "
                  "junction {j_start:4.0f} -> peak {j_peak:4.0f}C  mem {m_peak:4.0f}C  "
                  "{p_peak:5.1f}W  fan {fan} rpm".format(n=len(bodies), **row), flush=True)
            if args.gap and ch < args.chapters and not guard.tripped.is_set():
                time.sleep(args.gap)
    except KeyboardInterrupt:
        print("\ninterrupted — stopping submission", flush=True)
    finally:
        guard.stop_flag.set()
        guard.join(timeout=5)

    after = sensors.read()
    print(f"\nlog: {args.log}   (junction now {after['junction']:.0f}C)")
    peaks = [r["j_peak"] for r in rows if r["j_peak"] is not None]
    # Over every sample, drain included — this is the number that has to clear 95.
    all_j = [d["junction"] for _, d in guard.samples if d["junction"] is not None]
    if all_j:
        print(f"observed maximum junction: {max(all_j):.0f}C  (ceiling 95C, "
              f"headroom {95 - max(all_j):.0f}C)")
    if guard.tripped.is_set():
        print(f"VERDICT: no safe plateau — cut fired ({guard.reason}) "
              f"after {len(rows)} chapter(s).")
    elif len(peaks) >= 3:
        climb = peaks[-1] - peaks[0]
        tail = max(peaks[-3:]) - min(peaks[-3:])
        if tail <= 2.0:
            print(f"VERDICT: plateau at ~{peaks[-1]:.0f}C — last 3 chapters within "
                  f"{tail:.0f}C, {climb:+.0f}C over the whole run. Headroom to 95C: "
                  f"{95 - max(peaks):.0f}C.")
        else:
            print(f"VERDICT: still climbing — {climb:+.0f}C across the run, last 3 span "
                  f"{tail:.0f}C. Run more chapters to find where it settles.")
    else:
        print("VERDICT: too few chapters to judge a plateau.")
    print(json.dumps(rows, ensure_ascii=False, default=lambda o: round(o, 1)
                     if isinstance(o, float) else o))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
