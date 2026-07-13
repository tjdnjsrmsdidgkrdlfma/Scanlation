#!/usr/bin/env python3
"""TEMP bench (revert via git): measure recognize-pool WORKER OCCUPANCY across a
``--parallel`` recognize-only batch — the direct test of the "gate+K is an image
bundle, so workers idle when a page's crops sum < W" claim, and whether erasing image
boundaries (a crop queue / stage separation) would actually speed things up.

For each K it: sets the active recognizer's ``gpu_concurrency`` to K, resets the
server's occupancy sink, fires every image at once (skip_translate, force), then reads
two numbers that together answer "how much / how meaningful":

  * wall_ms  = max(total_ms) over the images = batch wall clock. Since all requests
               start together, the last to finish IS the end-to-end time. This is the
               REALIZED-speed metric (includes GPU time-slicing). K -> image-count
               approximates stage separation, so watching this plateau bounds the gain.
  * util     = avg busy workers / W = how full the pool was. 1.0 = workers were NEVER
               starved for crops (image boundaries cost ~nothing -> stage separation
               has nothing to recover); <1.0 is the worker capacity lost to starvation.
               Robust to time-slicing (a worker on a slow shared-GPU crop still counts
               as busy — only true starvation reads as idle). Plus a histogram of the
               % of wall time with exactly c workers busy, showing the idle's shape.

If util is already ~1.0 at the current default K, the idle the section worries about
isn't there and stage separation is moot — no need to build it to find out.

  # sweep K on the running GPU server; compare wall vs util down the rows
  python tools/bench_occupancy.py /path/to/chapter --server http://127.0.0.1:4010 --k 2 4 8 16 21

Requires the matching temp instrumentation (app/recognize_pool.py) + routes
(app/routes/admin.py) on this branch, and a server running the GPU recognizer (a real
pool with W>1). With the CPU in-process recognizer there is no pool and crops=0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse the running-server helpers from the sibling report tool (stdlib-only, same dir).
from run_report import _headers, get_settings, post_pipeline
from run_image import iter_images


def _req(server: str, path: str, token: str, *, method: str = "GET", body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"{server}{path}", data=data, method=method,
                                 headers=_headers(token, json_body=body is not None))
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _fire_parallel(server: str, token: str, imgs: list, skip_translate: bool) -> tuple[int, int, float | None]:
    """Fire every image at once (like a page load); return (ok, failed, wall_ms) where
    wall_ms = max(total_ms) over the successful timed responses."""
    ok, failed, totals = 0, 0, []
    with ThreadPoolExecutor(max_workers=len(imgs)) as ex:
        futs = [ex.submit(post_pipeline, server, token, img, True, skip_translate) for img in imgs]
        for f in as_completed(futs):
            try:
                resp = f.result()
            except Exception:  # noqa: BLE001 - one bad image shouldn't sink the batch
                failed += 1
                continue
            ok += 1
            t = (resp.get("timing") or {}).get("total_ms")
            if t is not None:
                totals.append(t)
    return ok, failed, (max(totals) if totals else None)


def _measure(server: str, token: str, imgs: list, k: int | None, rec: str | None) -> dict:
    if k is not None:
        if not rec:
            raise SystemExit("cannot set K: no active recognizer found in /get_settings/")
        _req(server, "/set_gpu_concurrency/", token, method="POST",
             body={"engine": rec, "concurrency": k})
    _req(server, "/bench_occupancy_reset/", token, method="POST")
    ok, failed, wall = _fire_parallel(server, token, imgs, skip_translate=True)
    occ = _req(server, "/bench_occupancy/", token)
    return {"k": k, "ok": ok, "failed": failed, "wall_ms": wall, "occ": occ}


def _fmt(row: dict) -> str:
    occ = row["occ"]
    k = "-" if row["k"] is None else row["k"]
    wall = f"{row['wall_ms']:.0f}ms" if row["wall_ms"] is not None else "n/a"
    if not occ.get("crops"):
        return f"K={k:>3} | imgs ok {row['ok']} fail {row['failed']} | wall {wall} | pool: no crops (CPU recognizer? no pool)"
    hist = "  ".join(f"{c}:{p:.0f}" for c, p in occ["pct_wall_by_busy_count"].items())
    util = occ["utilization"]
    return (f"K={k:>3} | imgs ok {row['ok']} fail {row['failed']} | wall {wall} "
            f"| W{occ['workers']} crops {occ['crops']} util {util:.2f} "
            f"avg_busy {occ['avg_busy_workers']:.2f} | busy% {hist}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure recognize-pool worker occupancy (+ wall clock) across a K sweep.")
    ap.add_argument("images", nargs="+", help="image files and/or folders")
    ap.add_argument("--server", default=os.environ.get("SCANLATION_SERVER", "http://127.0.0.1:4010"))
    ap.add_argument("--token", default=os.environ.get("SCANLATION_AUTH_TOKEN", ""))
    ap.add_argument("--k", nargs="*", type=int, default=None,
                    help="gpu_concurrency values to sweep (sets each on the active recognizer). "
                         "Omit to measure once at the server's current K.")
    ap.add_argument("--out", default=None, help="also write the rows as JSON to this path")
    a = ap.parse_args()

    server = a.server.rstrip("/")
    imgs = list(iter_images(a.images))
    if not imgs:
        print("no images found", file=sys.stderr)
        return 1

    rec = None
    if a.k:
        settings = get_settings(server, a.token) or {}
        rec = (settings.get("selection") or {}).get("recognizer")
    ks: list = a.k if a.k else [None]

    print(f"server {server}  ·  {len(imgs)} images  ·  recognizer {rec or '(current)'}", file=sys.stderr)
    rows = []
    for k in ks:
        row = _measure(server, a.token, imgs, k, rec)
        rows.append(row)
        print(_fmt(row))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)
        print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
