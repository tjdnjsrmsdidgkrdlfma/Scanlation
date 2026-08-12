#!/usr/bin/env python3
"""Cooling A/B — compare shroud configurations by coupling, not by steady state.

cooling-mi50-fans.md Task 3 asks for a fixed-rpm run held to steady state and
logged for ten minutes per configuration. That procedure predates the sustained
measurement and can no longer be run: at 150W the bare fan reaches junction 96C
in 50s and the climb never bends, so there is no steady state to reach and a
ten-minute log ends in a thermal cut inside the first minute.

What still separates a good configuration from a bad one is how fast heat leaves
the card, and that is measurable long before the ceiling:

    heat-up   seconds to climb gate-lo -> gate-hi under a fixed load and a fixed
              fan duty, reported as C/s. Lower is better coupling.
    cooldown  the time constant of the SLOW tail after submission stops, at the
              same fixed duty. Junction falls tens of degrees in the first
              seconds, but that is the die-to-sink gradient closing rather than
              heat leaving the card; the tail that follows is the assembly
              cooling through the fins, and that is the one worth timing.

Both metrics turn at 80C, fifteen degrees under the ceiling, and each rep costs
under a minute. They answer the question Task 3 was really asking -- does the
shroud move air through the fins -- without ever asking the card to survive a
load it cannot survive.

Run the same command once per configuration, changing only --label:

    python tools/bench_cooling_ab.py report.md --label bare
    python tools/bench_cooling_ab.py report.md --label shroud2
    python tools/bench_cooling_ab.py report.md --label shroud3
    python tools/bench_cooling_ab.py --compare cooling_bare.json cooling_shroud2.json

A cardboard mock-up counts as a configuration. Taping over the uncovered part of
the card's intake aperture is a five-minute test of the bypass hypothesis, and it
is worth running before committing to a printed shroud.

--sweep answers the same question without touching the hardware at all, and gives
absolute numbers rather than a difference between two configurations:

    python tools/bench_cooling_ab.py report.md --label bare --sweep 26,51,77,102,128

The cooldown is a three-parameter exponential, so fitting it yields the time
constant AND the intake temperature this board has no sensor for. Feeding that
back into the lumped model

    C dT/dt = P - (T - T_amb)/R_th        with R_th = tau / C

pins the heat capacity from the heat-up window, and with it the thermal
resistance and the watts actually leaving the card at the gate. Repeating that at
several duties gives R_th against rpm, which is the objective form of the
question: a stack the air passes through sheds resistance roughly as V^-0.65,
while a stack the air goes around barely moves. Flat means the leak sets the
resistance, and no fan speed will fix it.

Treat R_th as a single-lump approximation -- early in a transient the die heats
before the fin mass does, so the fitted capacity is smaller than the sink's true
one. The SHAPE across duties is the robust part, and it is the part that decides.

FAN CONTROL: this tool pins pwm4 for the duration, which means stopping
fancontrol -- it rewrites the channel every INTERVAL and would otherwise pull the
duty back mid-measurement. Control is handed back unconditionally on exit,
including on SIGINT/SIGTERM. The power cap is never touched.

Safety is bench_translate_sustained's Guard, unchanged. The gates here stop the
load 5C below its lowest cut, so the guard is a backstop that should not fire; if
it does, the configuration under test is worse than the bare fan.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 - UTF-8 stdio

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import bench_translate_concurrency as btc
import bench_translate_sustained as bts

# Backstop cuts, mirroring bench_translate_sustained. The gates stop the load far
# below these, so they exist to catch a configuration that heats faster than any
# measured so far -- not to shape the measurement.
GUARD = dict(cut=85.0, cut_hard=88.0, cut_project=91.0, cut_mem=80.0,
             interval=0.5, slope_window=5.0, horizon=3.0,
             # bench_translate_sustained arms the projection at 78C because it
             # intends to keep climbing. This tool stops the load at the gate, so
             # arming below the gate makes the projection fire on the approach to
             # a temperature we were going to stop at anyway -- it has to sit
             # above the gate, where only the drain overshoot can reach it.
             project_arm=84.0)

# Under load the card draws 130-170W. hwmon intermittently returns an idle-time
# block instead -- temperature 10-20C low with power near 23W -- and those samples
# would flatten the slope they land in. Power is the discriminator, so drop the
# sample when it is implausibly low for a loaded card.
LOAD_POWER_FLOOR = 60.0


def _write_sysfs(path: str, value: int) -> None:
    try:
        with open(path, "w") as f:
            f.write(str(value))
    except OSError as e:
        sys.exit(f"cannot write {path}: {e} — run as root?")


def _read_int(path: str) -> int | None:
    return bts._read_int(path)


def _slope(samples, t_from: float, t_to: float) -> float | None:
    """Least-squares C/s over a window, ignoring the idle-block samples."""
    pts = [(t, d["junction"]) for t, d in samples
           if t_from <= t <= t_to and d["junction"] is not None
           and (d["power"] or 0) >= LOAD_POWER_FLOOR]
    if len(pts) < 4:
        return None
    n = len(pts)
    mt = sum(t for t, _ in pts) / n
    mj = sum(j for _, j in pts) / n
    denom = sum((t - mt) ** 2 for t, _ in pts)
    if denom <= 0:
        return None
    return sum((t - mt) * (j - mj) for t, j in pts) / denom


def _fit_cooldown(samples, t_from: float, t_to: float):
    """Fit T(t) = T_amb + (T0 - T_amb) * exp(-t/tau) to a fan-on cooldown.

    Three parameters and no scipy: for any candidate ambient the fit is linear in
    log space, so scanning ambient and keeping the smallest residual pins all
    three. Intake temperature falls out of the curve, which is what makes the
    absolute numbers possible -- there is no intake sensor on this board.

    Returns (tau_s, t_amb_c, rmse_c).
    """
    pts = [(t - t_from, d["junction"]) for t, d in samples
           if t_from <= t <= t_to and d["junction"] is not None]
    if len(pts) < 8:
        return None, None, None
    t_last = pts[-1][1]
    best = None
    amb = 5.0
    while amb <= t_last - 2.0:
        ys = []
        for t, temp in pts:
            gap = temp - amb
            if gap <= 0.5:
                ys = []
                break
            ys.append((t, math.log(gap)))
        if len(ys) >= 8:
            n = len(ys)
            mt = sum(t for t, _ in ys) / n
            my = sum(y for _, y in ys) / n
            den = sum((t - mt) ** 2 for t, _ in ys)
            if den > 0:
                b = sum((t - mt) * (y - my) for t, y in ys) / den
                a = my - b * mt
                if b < 0:
                    rmse = math.sqrt(
                        sum((amb + math.exp(a + b * t) - temp) ** 2 for t, temp in pts) / n)
                    if best is None or rmse < best[2]:
                        best = (-1.0 / b, amb, rmse)
        amb += 0.5
    return best if best else (None, None, None)


def _slope_any(samples, t_from: float, t_to: float) -> float | None:
    """Least-squares C/s with no power filter — valid for an idle, cooling card."""
    pts = [(t, d["junction"]) for t, d in samples
           if t_from <= t <= t_to and d["junction"] is not None]
    if len(pts) < 5:
        return None
    n = len(pts)
    mt = sum(t for t, _ in pts) / n
    mj = sum(j for _, j in pts) / n
    den = sum((t - mt) ** 2 for t, _ in pts)
    return sum((t - mt) * (j - mj) for t, j in pts) / den if den > 0 else None


def _mean_load(samples, t_from: float, t_to: float):
    """Mean power and junction over a heat-up window, idle-block samples dropped."""
    rows = [d for t, d in samples if t_from <= t <= t_to
            and d["junction"] is not None and (d["power"] or 0) >= LOAD_POWER_FLOOR]
    if not rows:
        return None, None
    return (sum(r["power"] for r in rows) / len(rows),
            sum(r["junction"] for r in rows) / len(rows))


def _mean_rpm(samples, t_from: float, t_to: float) -> int | None:
    vals = [d["fan_rpm"] for t, d in samples
            if t_from <= t <= t_to and d["fan_rpm"] is not None]
    return round(sum(vals) / len(vals)) if vals else None


def _median(xs: list[float]) -> float | None:
    vals = sorted(x for x in xs if x is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


class FixedFan:
    """Pins pwm4 to one duty for the run, then hands the channel back.

    fancontrol has to be stopped rather than outvoted: it rewrites pwm4 every
    INTERVAL seconds, so a duty written underneath it survives about five seconds.
    Restore runs from a finally block and from the signal handlers, because a
    stale manual duty is the one failure mode that outlives the process.
    """

    def __init__(self, fan_dir: str, duty: int, unit: str = "fancontrol"):
        self.fan = fan_dir
        self.duty = duty
        self.unit = unit
        self.was_active = False
        self.prev_enable: int | None = None
        self.prev_duty: int | None = None
        self.restored = False
        self.stop_hold = threading.Event()
        self.holder: threading.Thread | None = None

    def __enter__(self) -> "FixedFan":
        self.was_active = subprocess.run(
            ["systemctl", "is-active", "--quiet", self.unit]).returncode == 0
        self.prev_enable = _read_int(self.fan + "pwm4_enable")
        self.prev_duty = _read_int(self.fan + "pwm4")
        if self.was_active:
            subprocess.run(["systemctl", "stop", self.unit], check=True)
        _write_sysfs(self.fan + "pwm4_enable", 1)
        _write_sysfs(self.fan + "pwm4", self.duty)
        # Stopping fancontrol hands the channel straight back to the EC, which
        # ramps pwm4 off its own CPU curve within a second. One write does not
        # hold it: rewriting on a short period is exactly what fancontrol does,
        # and without it the duty under measurement is not the duty requested.
        self.holder = threading.Thread(target=self._hold, daemon=True)
        self.holder.start()
        return self

    def _hold(self) -> None:
        while not self.stop_hold.wait(1.0):
            try:
                with open(self.fan + "pwm4", "w") as f:
                    f.write(str(self.duty))
            except OSError:
                return

    def set_duty(self, duty: int, settle: float = 0.0, tol: float = 0.03,
                 timeout: float = 120.0) -> int | None:
        """Write a duty and wait until the tachometer actually settles on it.

        The EC follows an increase instantly but walks a decrease down over tens
        of seconds, so a fixed sleep records a speed the fan is still leaving. Wait
        for five consecutive readings inside `tol` instead, and return that speed.
        """
        self.duty = duty
        _write_sysfs(self.fan + "pwm4", duty)
        deadline = time.monotonic() + timeout
        recent: list[int] = []
        while time.monotonic() < deadline:
            time.sleep(2.0)
            rpm = _read_int(self.fan + "fan4_input")
            if rpm is None:
                continue
            recent = (recent + [rpm])[-5:]
            if len(recent) == 5 and min(recent) > 0 and \
                    (max(recent) - min(recent)) / min(recent) <= tol:
                break
        if settle:
            time.sleep(settle)
        return _read_int(self.fan + "fan4_input")

    def __exit__(self, *exc) -> None:
        self.restore()

    def restore(self) -> None:
        if self.restored:
            return
        self.restored = True
        if self.was_active:
            # fancontrol claims the channel itself on start, so restarting it is a
            # truer restore than guessing an enable value.
            subprocess.run(["systemctl", "start", self.unit], check=False)
        else:
            if self.prev_duty is not None:
                _write_sysfs(self.fan + "pwm4", self.prev_duty)
            if self.prev_enable is not None:
                _write_sysfs(self.fan + "pwm4_enable", self.prev_enable)
        print(f"fan control returned to {'fancontrol' if self.was_active else 'previous state'}",
              flush=True)


class PowerCap:
    """Temporarily lowers the GPU power cap, restoring it unconditionally.

    The point of lowering it is to make a steady state exist. At 150W the card
    has no equilibrium inside the safe range, so R_th can only be inferred
    through a model; at 60W it settles, and R_th = (junction - intake)/P is read
    straight off the card with no heat capacity and no curve fitting involved.
    """

    def __init__(self, gpu_dir: str, watts: float):
        self.gpu = gpu_dir
        self.watts = watts
        self.prev: int | None = None
        self.restored = False

    def __enter__(self) -> "PowerCap":
        self.prev = _read_int(self.gpu + "power1_cap")
        lo = _read_int(self.gpu + "power1_cap_min") or 0
        hi = _read_int(self.gpu + "power1_cap_max") or 0
        target = int(self.watts * 1_000_000)
        if hi and target > hi or (lo and target < lo):
            sys.exit(f"cap {self.watts}W is outside the card's range "
                     f"{lo / 1e6:.0f}-{hi / 1e6:.0f}W")
        _write_sysfs(self.gpu + "power1_cap", target)
        print(f"power cap {(self.prev or 0) / 1e6:.0f}W -> {self.watts:.0f}W", flush=True)
        return self

    def __exit__(self, *exc) -> None:
        self.restore()

    def restore(self) -> None:
        if self.restored or self.prev is None:
            return
        self.restored = True
        _write_sysfs(self.gpu + "power1_cap", self.prev)
        print(f"power cap restored to {self.prev / 1e6:.0f}W", flush=True)


class Loader:
    """Keeps `concurrency` requests in flight, cycling the chapter's pages."""

    def __init__(self, url: str, bodies: list, concurrency: int, timeout: float, guard):
        self.url = url
        self.bodies = bodies
        self.timeout = timeout
        self.guard = guard
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.next = 0
        self.threads = [threading.Thread(target=self._worker, daemon=True)
                        for _ in range(concurrency)]

    def _worker(self) -> None:
        while not self.stop.is_set() and not self.guard.tripped.is_set():
            with self.lock:
                body = self.bodies[self.next % len(self.bodies)]
                self.next += 1
            try:
                btc._post(self.url, body, self.timeout)
            except Exception:
                # A failed request is not a reason to abandon a thermal run; the
                # load simply carries on with the next page.
                pass

    def start(self) -> None:
        for t in self.threads:
            t.start()

    def drain(self, timeout: float = 180.0) -> None:
        """Stop submitting and let in-flight requests finish on their own.

        Never signals llama-server: killing a process mid-GPU-work wedges the
        amdgpu context and takes a cold boot to clear.
        """
        self.stop.set()
        end = time.monotonic() + timeout
        for t in self.threads:
            t.join(timeout=max(0.0, end - time.monotonic()))


def _latest(guard):
    return guard.samples[-1][1] if guard.samples else None


def _wait_above(guard, target: float, timeout: float) -> float | None:
    """Time of the first sample at or above `target`, or None on timeout/trip."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if guard.tripped.is_set():
            return None
        for t, d in reversed(guard.samples[-4:]):
            j = d["junction"]
            if j is not None and j >= target:
                return t
        time.sleep(0.2)
    return None


def _wait_below(guard, target: float, timeout: float) -> float | None:
    """Time of the first of two consecutive samples at or below `target`.

    Two samples because the idle-block artefact reads low, and on the way down
    the power floor cannot filter it -- an idle card really is drawing 23W.
    """
    end = time.monotonic() + timeout
    seen = 0
    last_t = None
    while time.monotonic() < end:
        if guard.tripped.is_set():
            return None
        d = _latest(guard)
        if d and d["junction"] is not None and d["junction"] <= target:
            seen += 1
            if seen >= 2:
                return last_t if last_t is not None else time.monotonic()
            last_t = guard.samples[-1][0]
        else:
            seen = 0
            last_t = None
        time.sleep(0.25)
    return None


def _settle(guard, target: float, timeout: float, soak_s: float, soak_rate: float) -> bool:
    """Wait for junction <= target AND for the curve to go flat.

    Junction is the die: it sheds tens of degrees in seconds while the fin mass is
    still holding the last run's heat. Starting on junction alone would hand
    whichever duty ran first a colder heatsink and make the sweep unfair, so also
    require the fall to stop -- that is the mass, not the die, reaching
    equilibrium with the fan.
    """
    print(f"  settling to junction <= {target:.0f}C, flat within "
          f"{soak_rate:.2f} C/s over {soak_s:.0f}s ...", end="", flush=True)
    deadline = time.monotonic() + timeout
    if _wait_below(guard, target, timeout) is None:
        print(" timeout", flush=True)
        return False
    while time.monotonic() < deadline:
        now = time.monotonic()
        rate = _slope_any(guard.samples, now - soak_s, now)
        d = _latest(guard)
        if rate is not None and abs(rate) <= soak_rate and d and \
                d["junction"] is not None and d["junction"] <= target:
            print(f" {d['junction']:.0f}C ({rate:+.3f} C/s)", flush=True)
            return True
        time.sleep(2.0)
    print(" timeout waiting for flat", flush=True)
    return False


def run_rep(rep: int, args, guard, url, bodies) -> dict | None:
    if not _settle(guard, args.settle_to, args.settle_timeout, args.soak_s, args.soak_rate):
        print(f"  rep {rep}: card never settled to {args.settle_to}C — skipping", flush=True)
        return None

    loader = Loader(url, bodies, args.concurrency, args.timeout, guard)
    t_start = time.monotonic()
    loader.start()

    t_lo = _wait_above(guard, args.gate_lo, args.heat_timeout)
    if t_lo is None:
        loader.drain()
        print(f"  rep {rep}: never reached {args.gate_lo:.0f}C — load too light?", flush=True)
        return None
    t_hi = _wait_above(guard, args.gate_hi, args.heat_timeout)
    loader.drain()
    if t_hi is None:
        print(f"  rep {rep}: never reached {args.gate_hi:.0f}C", flush=True)
        return None

    heat_s = t_hi - t_lo
    slope = _slope(guard.samples, t_lo, t_hi)

    # Cooldown is timed from the post-drain peak, not from the moment submission
    # stopped: in-flight work keeps heating the card for a second or two and that
    # overshoot belongs to neither metric.
    time.sleep(2.0)
    tail = [(t, d["junction"]) for t, d in guard.samples
            if t >= t_hi and d["junction"] is not None]
    t_peak, j_peak = max(tail, key=lambda p: p[1]) if tail else (t_hi, args.gate_hi)

    # Junction sheds tens of degrees in the first seconds after submission stops.
    # That is the die-to-sink gradient closing, not the sink losing heat -- fitting
    # it returns the die's time constant, which is not the one cooling is limited
    # by. Skip that transient and fit the slow tail, where the whole assembly is
    # cooling together and tau actually describes the path to the air.
    t_fit_from = t_peak + args.cool_skip
    t_fit_to = t_fit_from + args.cool_fit_s
    while time.monotonic() < t_fit_to and not guard.tripped.is_set():
        time.sleep(1.0)
    t_cool = next((t for t, d in guard.samples
                   if t >= t_peak and d["junction"] is not None
                   and d["junction"] <= args.cool_to), None)
    cool_s = (t_cool - t_peak) if t_cool is not None else None

    def at(offset: float) -> float | None:
        after = [d["junction"] for t, d in guard.samples
                 if t >= t_peak + offset and d["junction"] is not None]
        return after[0] if after else None

    # Absolute characterisation. The cooldown fit supplies tau and the intake
    # temperature the board has no sensor for; the lumped model
    #   C dT/dt = P - (T - T_amb)/R_th,  R_th = tau / C
    # then pins C from the heat-up window, and with it the watts this
    # configuration actually removes at the gate. That number stands on its own --
    # it does not need a second configuration to mean something.
    tau, t_amb, rmse = _fit_cooldown(guard.samples, t_fit_from, t_fit_to)
    p_mean, j_mean = _mean_load(guard.samples, t_lo, t_hi)
    cap_c = r_th = q_out = None
    if tau and t_amb is not None and slope and p_mean and j_mean is not None:
        denom = slope + (j_mean - t_amb) / tau
        if denom > 0:
            cap_c = p_mean / denom            # J/C
            r_th = tau / cap_c                # C/W
            q_out = (args.gate_hi - t_amb) / r_th   # W removed at the gate

    row = {
        "rep": rep,
        "heat_s": round(heat_s, 1),
        "slope_c_s": round(slope, 3) if slope is not None else None,
        "j_peak": round(j_peak, 1),
        "cool_s": round(cool_s, 1) if cool_s is not None else None,
        "j_plus10": at(10.0),
        "j_plus30": at(30.0),
        "ttfl_s": round(t_lo - t_start, 1),
        # The speed the fan actually held while the slope was being measured, not
        # the one it happened to show when the duty was written.
        "fan_rpm_meas": _mean_rpm(guard.samples, t_lo, t_hi),
        "power_w": round(p_mean, 1) if p_mean else None,
        "tau_s": round(tau, 1) if tau else None,
        "t_amb_c": round(t_amb, 1) if t_amb is not None else None,
        "fit_rmse_c": round(rmse, 2) if rmse else None,
        "heat_capacity_j_c": round(cap_c, 1) if cap_c else None,
        "r_th_c_w": round(r_th, 4) if r_th else None,
        "q_out_w": round(q_out, 1) if q_out else None,
    }
    print("  rep {rep}: {lo:.0f}->{hi:.0f}C in {heat_s:.1f}s ({slope} C/s @ {pw}W)  "
          "cool {cool}  tau {tau}  intake {amb}  ->  R_th {rth}  Q {q}".format(
              lo=args.gate_lo, hi=args.gate_hi, rep=rep, heat_s=heat_s,
              slope=f"{slope:.2f}" if slope is not None else "?",
              pw=f"{p_mean:.0f}" if p_mean else "?",
              cool=f"{cool_s:.0f}s" if cool_s is not None else "timeout",
              tau=f"{tau:.0f}s" if tau else "?",
              amb=f"{t_amb:.0f}C" if t_amb is not None else "?",
              rth=f"{r_th:.3f} C/W" if r_th else "?",
              q=f"{q_out:.0f}W" if q_out else "?"), flush=True)
    return row


def run_steady(duty: int, rpm, args, guard, url, bodies) -> dict | None:
    """Hold the load until junction stops moving, then read R_th off the card.

    No lumped model here: at a cap the card can actually shed, junction settles
    and R_th = (junction - intake) / P falls out of two measured numbers. This is
    the arbiter for the disagreement between tau and the modelled R_th.
    """
    loader = Loader(url, bodies, args.concurrency, args.timeout, guard)
    t0 = time.monotonic()
    loader.start()
    deadline = t0 + args.steady_timeout
    flat_since = None
    print(f"  holding load until flat within {args.steady_rate:.3f} C/s over "
          f"{args.steady_window:.0f}s ...", flush=True)
    try:
        while time.monotonic() < deadline and not guard.tripped.is_set():
            time.sleep(5.0)
            now = time.monotonic()
            if now - t0 < args.steady_window:
                continue
            # _slope, not _slope_any: the card is under load here, so hwmon's
            # idle-time blocks are identifiable and have to go. _slope_any exists
            # for the cooldown, where a low power reading is genuine.
            rate = _slope(guard.samples, now - args.steady_window, now)
            d = _latest(guard)
            if rate is None or d is None or d["junction"] is None:
                continue
            print(f"    {now - t0:5.0f}s  junction {d['junction']:5.1f}C  "
                  f"{rate:+.3f} C/s  {d['power'] or 0:5.1f}W", flush=True)
            # A rate threshold alone declares victory too early: junction wobbles
            # about 1.5C sample to sample, and one low reading at the end of the
            # window can drag the regression under the threshold while the card is
            # still climbing. Require real time under load as well -- an
            # exponential approach cannot be flat at half a time constant.
            if abs(rate) <= args.steady_rate and now - t0 >= args.steady_min_s:
                flat_since = now
                break
    finally:
        loader.drain()

    if guard.tripped.is_set():
        print(f"  backstop fired before settling — this duty's equilibrium is above "
              f"the cut at {args.steady_watts:.0f}W", flush=True)
        return None
    if flat_since is None:
        print(f"  did not settle within {args.steady_timeout:.0f}s", flush=True)
        return None

    # Every average here takes the same power filter. An idle-time block reads
    # junction 10-20C low as well as power low, so averaging temperature over
    # unfiltered samples drags the equilibrium down and R_th with it.
    win = [d for t, d in guard.samples
           if flat_since - args.steady_window <= t <= flat_since
           and (d["power"] or 0) >= LOAD_POWER_FLOOR]
    js = [d["junction"] for d in win if d["junction"] is not None]
    ps = [d["power"] for d in win if d["power"] is not None]
    ms = [d["mem"] for d in win if d["mem"] is not None]
    if not js or not ps:
        return None
    j_ss, p_ss = sum(js) / len(js), sum(ps) / len(ps)
    r_th = (j_ss - args.intake) / p_ss
    row = {
        "pwm": duty,
        "fan_rpm": _mean_rpm(guard.samples, flat_since - args.steady_window, flat_since),
        "settle_s": round(flat_since - t0, 1),
        "junction_ss": round(j_ss, 2),
        "mem_ss": round(sum(ms) / len(ms), 1) if ms else None,
        "power_ss": round(p_ss, 1),
        "intake_assumed": args.intake,
        "r_th_c_w": round(r_th, 4),
    }
    print(f"  settled in {row['settle_s']:.0f}s: junction {j_ss:.1f}C at {p_ss:.1f}W "
          f"-> R_th {r_th:.3f} C/W  (mem {row['mem_ss']}C)", flush=True)
    return row


MEDIAN_KEYS = ("heat_s", "slope_c_s", "cool_s", "j_plus10", "j_plus30", "power_w",
               "fan_rpm_meas", "tau_s", "t_amb_c", "heat_capacity_j_c", "r_th_c_w",
               "q_out_w")

# Forced convection over a fin stack puts the film coefficient near V^0.6-0.8, so
# a heatsink the air actually passes through sheds resistance at roughly this
# exponent. Series conduction inside the card damps it, which is why the printed
# expectation is a ceiling to compare against rather than a target to hit.
COUPLED_EXPONENT = 0.65


def print_sweep(levels: list[dict], args) -> None:
    print(f"\n{'pwm':>5}{'rpm':>8}{'C/s':>8}{'W':>7}{'tau s':>8}{'intake':>8}"
          f"{'R_th C/W':>11}{'Q@' + format(args.gate_hi, '.0f') + 'C':>10}")
    for lv in levels:
        m = lv["median"]

        def f(key, fmt):
            v = m.get(key)
            return format(v, fmt) if v is not None else "?"

        print(f"{lv['pwm']:>5}{lv['fan_rpm'] or '?':>8}{f('slope_c_s', '.2f'):>8}"
              f"{f('power_w', '.0f'):>7}{f('tau_s', '.0f'):>8}{f('t_amb_c', '.0f'):>8}"
              f"{f('r_th_c_w', '.3f'):>11}{f('q_out_w', '.0f'):>10}")

    pts = [(lv["fan_rpm"], lv["median"]["r_th_c_w"]) for lv in levels
           if lv["fan_rpm"] and lv["median"]["r_th_c_w"]]
    if len(pts) < 2:
        print("\nnot enough usable duties to judge coupling.")
        return
    (rpm_lo, r_lo), (rpm_hi, r_hi) = pts[0], pts[-1]
    ratio = rpm_hi / rpm_lo
    got = (1 - r_hi / r_lo) * 100
    want = (1 - ratio ** -COUPLED_EXPONENT) * 100
    print(f"\nfan {rpm_lo} -> {rpm_hi} rpm ({ratio:.1f}x): R_th {r_lo:.3f} -> {r_hi:.3f} C/W "
          f"({-got:+.0f}%). A stack the air passes through would give about -{want:.0f}%.")
    if got < want * 0.4:
        print("VERDICT: airflow is not reaching the fins. Resistance is set by the leak "
              "path, not by fan speed -- which is why rpm buys nothing. Sealing the "
              "aperture is the lever; more rpm is not.")
    elif got < want * 0.75:
        print("VERDICT: partially coupled. Some flow crosses the fins, but a leak path "
              "is taking a large share of it.")
    else:
        print("VERDICT: coupling is behaving normally for a ducted stack — resistance "
              "tracks airflow. If temperature is still the problem, the limit is the "
              "heatsink itself, not the path to it.")


def compare(paths: list[str]) -> int:
    runs = []
    for p in paths:
        try:
            runs.append(json.loads(Path(p).read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:
            sys.exit(f"cannot read {p}: {e}")
    print(f"{'config':<14}{'rpm':>7}{'slope C/s':>12}{'heat s':>9}{'cool s':>9}{'+30s C':>9}  reps")
    for r in runs:
        m = r["median"]
        print("{:<14}{:>7}{:>12}{:>9}{:>9}{:>9}  {}".format(
            r["label"], r.get("fan_rpm") or "?",
            f"{m['slope_c_s']:.2f}" if m["slope_c_s"] is not None else "?",
            f"{m['heat_s']:.1f}" if m["heat_s"] is not None else "?",
            f"{m['cool_s']:.0f}" if m["cool_s"] is not None else "?",
            f"{m['j_plus30']:.0f}" if m["j_plus30"] is not None else "?",
            len(r["reps"])))
    base = runs[0]["median"]["slope_c_s"]
    if base:
        print()
        for r in runs[1:]:
            s = r["median"]["slope_c_s"]
            if s:
                print(f"{r['label']} vs {runs[0]['label']}: slope {(s / base - 1) * 100:+.0f}% "
                      f"({'better' if s < base else 'worse'} coupling)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", nargs="?", help="run_report markdown with per-page source tables")
    ap.add_argument("--compare", nargs="+", metavar="JSON",
                    help="print a side-by-side of finished runs and exit")
    ap.add_argument("--label", help="configuration name, e.g. bare / mock / shroud2 / shroud3")
    ap.add_argument("--pwm", type=int, default=123,
                    help="fixed pwm4 duty for the run (default 123 ~ 8,500 rpm bare)")
    ap.add_argument("--sweep", default="",
                    help="characterise R_th at several duties instead, e.g. 26,51,77,102,128. "
                         "Needs no physical access: if resistance barely falls as the fan "
                         "triples, the air is bypassing the fins")
    ap.add_argument("--reps", type=int, default=None,
                    help="repetitions per duty (default 1 when sweeping, else 3)")
    ap.add_argument("--steady", action="store_true",
                    help="lower the power cap until a steady state exists, hold the load "
                         "until junction stops moving, and read R_th directly. Settles the "
                         "tau-vs-R_th disagreement without a lumped model")
    ap.add_argument("--steady-watts", type=float, default=60.0,
                    help="power cap for --steady (default 60). Pick one whose equilibrium "
                         "clears the backstop at every duty tested")
    ap.add_argument("--steady-rate", type=float, default=0.01,
                    help="C/s that counts as settled (default 0.01, about 3 tau)")
    ap.add_argument("--steady-window", type=float, default=60.0,
                    help="window the rate is measured over (default 60)")
    ap.add_argument("--steady-min-s", type=float, default=400.0,
                    help="hold the load at least this long before believing the rate "
                         "threshold, roughly 2 tau (default 400)")
    ap.add_argument("--steady-timeout", type=float, default=1200.0,
                    help="give up on a duty after this long (default 1200)")
    ap.add_argument("--intake", type=float, default=31.0,
                    help="intake temperature for R_th, recovered from idle equilibrium "
                         "rather than from the cooldown asymptote (default 31)")
    ap.add_argument("-P", "--concurrency", type=int, default=4,
                    help="in-flight requests, matching the production gate (default 4)")
    ap.add_argument("--gate-lo", type=float, default=70.0, help="slope window start (default 70)")
    ap.add_argument("--gate-hi", type=float, default=80.0,
                    help="slope window end; load stops here (default 80)")
    ap.add_argument("--cool-to", type=float, default=65.0,
                    help="reported cooldown target; not what tau is fitted on (default 65)")
    ap.add_argument("--cool-skip", type=float, default=15.0,
                    help="seconds of post-load transient to skip before fitting, so tau "
                         "describes the heatsink and not the die (default 15)")
    ap.add_argument("--cool-fit-s", type=float, default=120.0,
                    help="length of the cooldown window tau is fitted on (default 120)")
    ap.add_argument("--settle-to", type=float, default=62.0,
                    help="start each rep at or below this junction (default 62)")
    ap.add_argument("--soak-s", type=float, default=20.0,
                    help="window the fall must be flat over before a rep starts, so every "
                         "duty meets the same heatsink and not just the same die (default 20)")
    ap.add_argument("--soak-rate", type=float, default=0.05,
                    help="C/s that counts as flat (default 0.05)")
    ap.add_argument("--settle-timeout", type=float, default=420.0)
    ap.add_argument("--heat-timeout", type=float, default=180.0)
    ap.add_argument("--cool-timeout", type=float, default=300.0)
    ap.add_argument("--fan-settle", type=float, default=15.0,
                    help="seconds for the EC to converge on the written duty (default 15)")
    ap.add_argument("--src", default="ja")
    ap.add_argument("--dst", default="ko")
    ap.add_argument("--model", default="")
    ap.add_argument("--endpoint", default=os.getenv("LLAMACPP_ENDPOINT", "http://127.0.0.1:8080"))
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", default="", help="result json (default cooling_<label>.json)")
    ap.add_argument("--log", default="", help="sensor csv (default cooling_<label>.csv)")
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare)
    if not args.report or not args.label:
        sys.exit("need REPORT and --label (or --compare)")
    if args.gate_hi >= GUARD["cut"]:
        sys.exit(f"--gate-hi must stay below the {GUARD['cut']}C backstop cut")
    try:
        duties = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.pwm]
    except ValueError:
        sys.exit("--sweep takes comma-separated integers, e.g. 26,51,77,102,128")
    if not all(0 <= d <= 255 for d in duties):
        sys.exit("pwm duties must be 0-255")
    if args.reps is None:
        args.reps = 1 if args.sweep else 3
    if args.steady:
        # The floor exists to drop hwmon's idle-time blocks, which read ~23W. A
        # 60W cap puts real load samples right at the default, so scale it to the
        # cap or the filter would throw the whole run away.
        global LOAD_POWER_FLOOR
        LOAD_POWER_FLOOR = min(LOAD_POWER_FLOOR, args.steady_watts * 0.5)

    out = Path(args.out or f"cooling_{args.label}.json")
    log = Path(args.log or f"cooling_{args.label}.csv")

    pages = btc.parse_pages(Path(args.report))
    if not pages:
        sys.exit(f"no page tables found in {args.report}")
    body_args = types.SimpleNamespace(
        model=args.model, src=args.src, dst=args.dst, max_tokens=args.max_tokens)
    bodies = [btc._body(texts, body_args) for _, texts in pages]
    url = f"{args.endpoint.rstrip('/')}/v1/chat/completions"

    sensors = bts.Sensors()
    if not sensors.fan:
        sys.exit("no nct6687 hwmon — is the module loaded? (dmesg | grep brute force)")
    lim = sensors.limits()
    print(f"MI50 hwmon {sensors.gpu}")
    print(f"  limits: junction crit {lim['junction_crit']:.0f}C · mem crit {lim['mem_crit']:.0f}C "
          f"· power cap {lim['power_cap']:.0f}W"
          f"{f' (lowered to {args.steady_watts:.0f}W for this run)' if args.steady else ' (untouched)'}")
    print(f"  config: {args.label} · pwm4 {','.join(str(d) for d in duties)} · gates "
          f"{args.gate_lo:.0f}->{args.gate_hi:.0f}C · cool to {args.cool_to:.0f}C · "
          f"{args.reps} rep(s) each · P={args.concurrency}")
    print(f"  backstop: sustained {GUARD['cut']}C · spike {GUARD['cut_hard']}C · "
          f"projected {GUARD['cut_project']}C · mem {GUARD['cut_mem']}C (ceiling 95C)\n")

    guard = bts.Guard(sensors, types.SimpleNamespace(**GUARD), log)
    guard.start()

    fan = FixedFan(sensors.fan, duties[0])
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, lambda *_: sys.exit("signal — restoring fan control"))
        except (ValueError, OSError):
            pass

    levels: list[dict] = []
    steady: list[dict] = []
    cap = PowerCap(sensors.gpu, args.steady_watts) if args.steady else None
    try:
        with fan:
            if cap:
                with cap:
                    for duty in duties:
                        rpm = fan.set_duty(duty, settle=args.fan_settle)
                        print(f"pwm4 = {duty} -> {rpm} rpm", flush=True)
                        row = run_steady(duty, rpm, args, guard, url, bodies)
                        if row:
                            steady.append(row)
                        if guard.tripped.is_set():
                            # The Event cannot be cleared, so the remaining duties
                            # would be skipped silently. Stop and say so instead.
                            print("backstop latched — rerun the remaining duties",
                                  flush=True)
                            break
            else:
                for duty in duties:
                    rpm = fan.set_duty(duty, settle=args.fan_settle)
                    print(f"pwm4 = {duty} -> {rpm} rpm", flush=True)
                    reps: list[dict] = []
                    for rep in range(1, args.reps + 1):
                        if guard.tripped.is_set():
                            break
                        row = run_rep(rep, args, guard, url, bodies)
                        if row:
                            reps.append(row)
                    med = {k: _median([r[k] for r in reps]) for k in MEDIAN_KEYS}
                    levels.append({
                        "pwm": duty,
                        # Prefer the speed held during the measurement; the settle
                        # reading is only a fallback for a rep that produced nothing.
                        "fan_rpm": round(med["fan_rpm_meas"]) if med["fan_rpm_meas"] else rpm,
                        "fan_rpm_at_settle": rpm,
                        "reps": reps,
                        "median": med,
                    })
                    if guard.tripped.is_set():
                        break
    except KeyboardInterrupt:
        print("\ninterrupted — stopping submission", flush=True)
    finally:
        guard.stop_flag.set()
        guard.join(timeout=5)
        if cap:
            cap.restore()
        fan.restore()

    first = levels[0] if levels else {"pwm": duties[0], "fan_rpm": None, "reps": [],
                                      "median": {k: None for k in MEDIAN_KEYS}}
    result = {
        "label": args.label,
        "concurrency": args.concurrency,
        "gate_lo": args.gate_lo,
        "gate_hi": args.gate_hi,
        "cool_to": args.cool_to,
        "power_cap_w": round(lim["power_cap"]),
        "pages_per_chapter": len(bodies),
        "levels": levels,
        "steady_watts": args.steady_watts if args.steady else None,
        "steady": steady,
        # Mirrors of the first duty, so a single-configuration run stays directly
        # comparable with --compare without unwrapping the sweep.
        "pwm": first["pwm"],
        "fan_rpm": first["fan_rpm"],
        "reps": first["reps"],
        "median": first["median"],
        "tripped": guard.reason if guard.tripped.is_set() else "",
        "log": str(log),
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nlog: {log}   result: {out}")
    all_j = [d["junction"] for _, d in guard.samples if d["junction"] is not None]
    if all_j:
        print(f"observed maximum junction: {max(all_j):.0f}C  (ceiling 95C, "
              f"headroom {95 - max(all_j):.0f}C)")
    if guard.tripped.is_set():
        print(f"BACKSTOP FIRED ({guard.reason}) — this configuration heats faster than the "
              f"gates assume; treat its numbers as a lower bound.")
    if args.steady:
        print(f"\nsteady state at {args.steady_watts:.0f}W, intake {args.intake:.0f}C "
              f"(no model, no heat capacity):")
        print(f"{'pwm':>5}{'rpm':>8}{'settle s':>10}{'junction':>10}{'mem':>7}"
              f"{'W':>7}{'R_th C/W':>11}")
        for r in steady:
            print(f"{r['pwm']:>5}{r['fan_rpm'] or '?':>8}{r['settle_s']:>10.0f}"
                  f"{r['junction_ss']:>10.1f}{r['mem_ss'] or 0:>7.0f}"
                  f"{r['power_ss']:>7.1f}{r['r_th_c_w']:>11.3f}")
        if len(steady) >= 2:
            lo, hi = steady[0], steady[-1]
            print(f"\nfan {lo['fan_rpm']} -> {hi['fan_rpm']} rpm: R_th "
                  f"{lo['r_th_c_w']:.3f} -> {hi['r_th_c_w']:.3f} C/W "
                  f"({(hi['r_th_c_w'] / lo['r_th_c_w'] - 1) * 100:+.0f}%). "
                  f"These are read off the card -- where they disagree with the "
                  f"modelled R_th, these win.")
    elif len(duties) > 1:
        print_sweep(levels, args)
    elif first["median"]["slope_c_s"] is not None:
        m = first["median"]
        cool = f"{m['cool_s']:.0f}s" if m["cool_s"] is not None else "not reached"
        rth = f"{m['r_th_c_w']:.3f} C/W" if m["r_th_c_w"] is not None else "?"
        print(f"VERDICT[{args.label}]: {m['slope_c_s']:.2f} C/s over "
              f"{args.gate_lo:.0f}-{args.gate_hi:.0f}C at {first['fan_rpm']} rpm, "
              f"cooldown to {args.cool_to:.0f}C {cool}, R_th {rth}. "
              f"Compare configurations with --compare, or sweep duties with --sweep.")
    else:
        print("VERDICT: no usable rep — nothing to report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
