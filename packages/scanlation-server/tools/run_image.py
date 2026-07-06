#!/usr/bin/env python3
"""Drive the RUNNING server's /run_pipeline/ on local image(s) — a headless repro
for log diagnosis when the browser extension isn't handy (e.g. from the office).

This posts the exact same request the extension does, so it exercises the real
production path (detect -> recognize -> batch translate) and the server prints the
per-detection / batch-failure lines to its log. The actual diagnosis is in the
SERVER log (`docker compose logs -f server`, with the 동작-tab verbose toggle on);
this tool just triggers the run and prints a one-line-per-region summary.

``force`` is ON by default, so the same image set re-runs the full pipeline every
time (cache ignored) — no cache-clearing between repro runs. Pass --use-cache to
honor the cache instead.

Requests go one at a time by default; --concurrency N sends N in parallel, which
reproduces the concurrent-load failures the extension hits when a page loads many
images at once (the server's translate slots overrun -> timeouts -> batch fallback).

  # every image under a folder, forced fresh, against the local docker port
  python tools/run_image.py samples/

  # reproduce concurrent-load failures (many images at once, like real browsing)
  python tools/run_image.py samples/ --concurrency 6

  # specific files, remote server behind nginx, with the auth token
  python tools/run_image.py a.png b.png --server https://scan.example.com --token SECRET

Server URL: --server or SCANLATION_SERVER (default http://127.0.0.1:4010, the
docker-compose host port). Auth token: --token or SCANLATION_AUTH_TOKEN. Only the
Python 3 stdlib is used, so it runs anywhere without installing anything.
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
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def iter_images(paths: list[str]):
    """Expand each arg: a directory yields its images (recursively, sorted so the
    order is stable across repro runs); a file is used as-is."""
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
        else:
            yield p


def run_one(server: str, token: str, img: Path, force: bool) -> list[dict]:
    """POST one image to /run_pipeline/ and return the result list. md5 is computed
    over the base64 *string* (what the server verifies), matching the extension."""
    b64 = base64.b64encode(img.read_bytes()).decode("ascii")
    md5 = hashlib.md5(b64.encode("utf-8")).hexdigest()
    body = json.dumps({"md5": md5, "contents": b64, "force": force}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Auth-Token"] = token
    req = urllib.request.Request(f"{server}/run_pipeline/", data=body, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["result"]


def format_block(img: Path, result: list[dict], quiet: bool) -> str:
    """One image's summary as a single string so parallel workers don't interleave
    lines. The real diagnosis is in the server log; this is just a sanity view."""
    lines = [f"\n=== {img.name} — {len(result)} regions ==="]
    if not quiet:
        lines += [f"  {i} {it['bounds']}  {it['source']!r} -> {it['destination']!r}"
                  for i, it in enumerate(result)]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless /run_pipeline/ repro for log diagnosis.")
    ap.add_argument("images", nargs="+", help="image files and/or folders")
    ap.add_argument("--server", default=os.environ.get("SCANLATION_SERVER", "http://127.0.0.1:4010"),
                    help="server base URL (default: %(default)s)")
    ap.add_argument("--token", default=os.environ.get("SCANLATION_AUTH_TOKEN", ""),
                    help="X-Auth-Token if the server requires one")
    ap.add_argument("--use-cache", action="store_true",
                    help="honor the cache (default: force a fresh run each time)")
    ap.add_argument("--quiet", action="store_true", help="print only the region count per image")
    ap.add_argument("--concurrency", type=int, default=1, metavar="N",
                    help="send N requests in parallel (default 1 = sequential). >1 reproduces "
                         "concurrent-load failures like real browsing (many images at once)")
    a = ap.parse_args()

    server = a.server.rstrip("/")
    force = not a.use_cache
    imgs = list(iter_images(a.images))
    if not imgs:
        print("no images found", file=sys.stderr)
        return 1
    conc = max(1, a.concurrency)
    print(f"POST {server}/run_pipeline/  ({len(imgs)} images, force={force}, concurrency={conc})",
          file=sys.stderr)

    def work(img: Path):
        try:
            return img, run_one(server, a.token, img, force), None
        except urllib.error.HTTPError as e:
            return img, None, f"HTTP {e.code} {e.read().decode('utf-8', 'replace')}"
        except Exception as e:  # noqa: BLE001 - network/decoding; report and move on
            return img, None, f"ERROR {e}"

    failures = 0
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for fut in as_completed([ex.submit(work, img) for img in imgs]):
            img, result, err = fut.result()
            if err:
                print(f"{img.name}: {err}", file=sys.stderr)
                failures += 1
            else:
                print(format_block(img, result, a.quiet))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
