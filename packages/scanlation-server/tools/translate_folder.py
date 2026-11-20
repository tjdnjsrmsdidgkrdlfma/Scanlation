#!/usr/bin/env python3
"""Translate every image in a folder through the RUNNING server and save each page
with the translation drawn on it, the way the extension overlays it in the browser.

Pages go one at a time, in file order. Each region is drawn like the extension's
box: its bounds filled with 92% white under a faint 1px border, the text centred at
the largest whole-px font size whose wrapped lines fit (the extension's measured
binary search, floored at the server's min_font_size). Pages land in OUT (default:
`<folder>_ko` beside the input) under their own names and format, and every region's
`{bounds, source, destination}` goes to OUT/scanlation.json. A page already in OUT is
skipped, so a stopped run resumes where it left off.

The MI50 has no thermal throttle, so the run watches it (PCI 1002:66a1, hwmon
temp2/temp3 = junction/mem) the whole time: before each page it waits while
junction is at --pause-at or above until it falls to --resume-at, and junction or
mem at --abort-at on two samples in a row stops the run once the page in flight
returns. It never signals the server. A host without the card (a remote server)
needs --no-guard.

  # on the GPU host, against the docker-compose port
  python tools/translate_folder.py /data/chapter
  python tools/translate_folder.py chapter/ --out chapter_ko --quality 90

Server URL: --server or SCANLATION_SERVER (default http://127.0.0.1:4010). Auth token:
--token or SCANLATION_AUTH_TOKEN. Needs Pillow (built with AVIF for .avif pages) and a
font covering the target language: --font, else fontconfig's match for --font-family.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 - UTF-8 stdio

import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from run_image import iter_images, run_one

MI50_ID = ("0x1002", "0x66a1")

# Overlay look, from extension/content.css (.scanlation-box).
BOX_BG = (255, 255, 255, round(0.92 * 255))
BOX_BORDER = (0, 0, 0, round(0.25 * 255))
TEXT_COLOR = (0, 0, 0)
LINE_HEIGHT = 1.05
INSET = 2  # 1px border + 1px padding on each side

# No line break before these (closing punctuation, prolonged sound marks).
NO_BREAK_BEFORE = set(".,!?…‥~)]}>»」』）］｝〉》’”、。！？・ー〜～♡♥")


# ------------------------------------------------------------------ MI50 guard --

def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def find_mi50() -> str | None:
    """The MI50's PCI device dir, found by vendor:device (the hwmon index moves per boot)."""
    for dev in glob.glob("/sys/bus/pci/devices/*/"):
        if (_read(dev + "vendor"), _read(dev + "device")) == MI50_ID:
            return dev
    return None


class Guard(threading.Thread):
    """Samples junction/mem once a second. A runtime-suspended card is idle and cold,
    and reading its sensors would wake it, so it reads as None."""

    def __init__(self, dev: str, abort_at: float):
        super().__init__(daemon=True)
        self.dev = dev
        self.abort_at = abort_at
        self.junction: float | None = None
        self.mem: float | None = None
        self.peak = 0.0
        self.tripped = threading.Event()
        self.reason = ""

    def sample(self) -> tuple[float | None, float | None]:
        if _read(self.dev + "power_state") not in (None, "D0"):
            return None, None
        hw = glob.glob(self.dev + "hwmon/hwmon*/")
        if not hw:
            return None, None
        vals = []
        for name in ("temp2_input", "temp3_input"):
            raw = _read(hw[0] + name)
            vals.append(int(raw) / 1000.0 if raw and raw.lstrip("-").isdigit() else None)
        return vals[0], vals[1]

    def run(self) -> None:
        over = 0
        while not self.tripped.is_set():
            self.junction, self.mem = self.sample()
            hot = [v for v in (self.junction, self.mem) if v is not None]
            if hot:
                self.peak = max(self.peak, *hot)
            if any(v >= self.abort_at for v in hot):
                over += 1
                if over >= 2:
                    self.reason = f"junction {self.junction}C / mem {self.mem}C"
                    self.tripped.set()
            else:
                over = 0
            time.sleep(1.0)


def wait_cool(guard: Guard, pause_at: float, resume_at: float) -> None:
    j = guard.junction
    if j is None or j < pause_at:
        return
    print(f"  MI50 junction {j:.0f}C >= {pause_at:.0f}C, waiting for {resume_at:.0f}C",
          file=sys.stderr, flush=True)
    while not guard.tripped.is_set():
        j = guard.junction
        if j is None or j <= resume_at:
            return
        time.sleep(2.0)


# ------------------------------------------------------------------- drawing --

def load_font(path: str | None, family: str):
    """A size -> FreeTypeFont factory. A variable font is pinned to its Regular instance."""
    index = 0
    if not path:
        out = subprocess.run(["fc-match", "-f", "%{file}\n%{index}", family],
                             capture_output=True, text=True, check=True).stdout.split("\n")
        path, index = out[0], int(out[1] or 0) & 0xFFFF
    cache: dict[int, ImageFont.FreeTypeFont] = {}

    def at(size: int) -> ImageFont.FreeTypeFont:
        if size not in cache:
            f = ImageFont.truetype(path, size, index=index)
            try:
                f.set_variation_by_name("Regular")
            except (OSError, ValueError):
                pass  # not a variable font, or no instance by that name
            cache[size] = f
        return cache[size]

    return path, at


def _cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x1100 <= o <= 0x11FF or 0x2E80 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF
            or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF)


def _tokens(s: str) -> list[str]:
    """Split at the line-break opportunities a browser takes: after whitespace, and
    around each CJK/Hangul character, never before closing punctuation."""
    toks: list[str] = []
    cur = ""
    for ch in s:
        if ch.isspace():
            if cur and not cur.isspace():
                toks.append(cur)
                cur = ""
            cur += ch
            continue
        if cur and (cur.isspace() or (ch not in NO_BREAK_BEFORE and (_cjk(ch) or _cjk(cur[0])))):
            toks.append(cur)
            cur = ""
        cur += ch
    if cur:
        toks.append(cur)
    return toks


def wrap(text: str, font, width: float) -> list[str]:
    """Greedy wrap like `white-space: pre-wrap; word-break: break-word`: newlines are
    kept, a token that cannot fit alone on a line is broken between characters."""
    lines: list[str] = []
    for para in text.split("\n"):
        line = ""
        for tok in _tokens(para):
            if tok.isspace():
                if line:
                    line += tok
                continue
            if line.strip() and font.getlength(line + tok) > width:
                lines.append(line.rstrip())
                line = ""
            if not line and font.getlength(tok) > width:
                for ch in tok:
                    if line and font.getlength(line + ch) > width:
                        lines.append(line)
                        line = ""
                    line += ch
            else:
                line += tok
        lines.append(line.rstrip())
    return lines


def fit(text: str, font_at, w: int, h: int, floor: int) -> tuple[int, list[str]]:
    """Largest whole-px size in [floor, box height] whose wrapped text fits the box
    (content area = box minus border and padding). Fits nothing -> the floor, clipped."""
    cw, ch = w - 2 * INSET, h - 2 * INSET

    def fits(size: int) -> bool:
        f = font_at(size)
        lines = wrap(text, f, cw)
        return (len(lines) * size * LINE_HEIGHT <= ch
                and all(f.getlength(ln) <= cw for ln in lines))

    lo, hi = floor, max(floor, h - 2)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo, wrap(text, font_at(lo), cw)


def draw_page(img: Image.Image, result: list[dict], font_at, floor: int) -> Image.Image:
    page = img.convert("RGB")
    W, H = page.size
    for item in result:
        l, t, r, b = item["bounds"]
        l, t = max(0, round(l)), max(0, round(t))
        r, b = min(W, round(r)), min(H, round(b))
        w, h = r - l, b - t
        if w <= 2 * INSET or h <= 2 * INSET:
            continue
        # The box is its own tile, so overflowing text clips at its edge (overflow: hidden).
        tile = page.crop((l, t, r, b)).convert("RGBA")
        tile = Image.alpha_composite(tile, Image.new("RGBA", (w, h), BOX_BG))
        edge = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(edge).rectangle((0, 0, w - 1, h - 1), outline=BOX_BORDER)
        tile = Image.alpha_composite(tile, edge).convert("RGB")

        text = item.get("destination") or ""
        size, lines = fit(text, font_at, w, h, floor)
        f = font_at(size)
        asc, desc = f.getmetrics()
        lh = size * LINE_HEIGHT
        y = (h - len(lines) * lh) / 2
        d = ImageDraw.Draw(tile)
        for ln in lines:
            # CSS centres the font's ascent+descent inside each line box (half-leading).
            baseline = y + (lh - (asc + desc)) / 2 + asc
            d.text(((w - f.getlength(ln)) / 2, baseline), ln, font=f, fill=TEXT_COLOR, anchor="ls")
            y += lh
        page.paste(tile, (l, t))
    return page


def save(page: Image.Image, path: Path, quality: int) -> None:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg", ".webp", ".avif"):
        page.save(path, quality=quality)
    else:
        page.save(path)


# ---------------------------------------------------------------------- main --

def handshake(server: str, token: str) -> dict:
    req = urllib.request.Request(f"{server}/", headers={"X-Auth-Token": token} if token else {})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main() -> int:
    ap = argparse.ArgumentParser(description="Translate a folder of pages and save them with the overlay drawn on.")
    ap.add_argument("folder", type=Path, help="folder of page images (read recursively, file order)")
    ap.add_argument("--out", type=Path, help="output folder (default: <folder>_ko beside it)")
    ap.add_argument("--server", default=os.environ.get("SCANLATION_SERVER", "http://127.0.0.1:4010"))
    ap.add_argument("--token", default=os.environ.get("SCANLATION_AUTH_TOKEN", ""))
    ap.add_argument("--font", help="font file (default: fontconfig's match for --font-family)")
    ap.add_argument("--font-family", default="Noto Sans CJK KR")
    ap.add_argument("--min-font-size", type=int, help="font-size floor px (default: the server's min_font_size)")
    ap.add_argument("--quality", type=int, default=90, help="JPEG/WebP/AVIF quality (default: %(default)s)")
    ap.add_argument("--overwrite", action="store_true", help="redo pages already in the output folder")
    ap.add_argument("--pause-at", type=float, default=80.0, help="MI50 junction C that holds the next page")
    ap.add_argument("--resume-at", type=float, default=65.0, help="junction C a held run resumes at")
    ap.add_argument("--abort-at", type=float, default=90.0, help="junction/mem C that stops the run")
    ap.add_argument("--no-guard", action="store_true", help="run without the MI50 on this host")
    a = ap.parse_args()

    src = a.folder.resolve()
    out = (a.out or src.with_name(src.name + "_ko")).resolve()
    pages = [p for p in iter_images([str(src)]) if out not in p.resolve().parents]
    if not pages:
        print("no images found", file=sys.stderr)
        return 1
    server = a.server.rstrip("/")
    floor = a.min_font_size or int(handshake(server, a.token).get("min_font_size") or 7)
    font_path, font_at = load_font(a.font, a.font_family)

    guard = None
    if not a.no_guard:
        dev = find_mi50()
        if not dev:
            sys.exit("no MI50 (1002:66a1) on this host; run on the GPU host or pass --no-guard")
        guard = Guard(dev, a.abort_at)
        guard.start()
        time.sleep(1.2)  # first sample in before the first page

    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "scanlation.json"
    index = json.loads(index_path.read_text("utf-8")) if index_path.exists() else {}
    print(f"{len(pages)} pages  {src} -> {out}  font {Path(font_path).name} floor {floor}px",
          file=sys.stderr, flush=True)

    failures = []
    t_run = time.monotonic()
    for i, page in enumerate(pages, 1):
        rel = page.relative_to(src)
        dst = out / rel
        if dst.exists() and not a.overwrite:
            continue
        if guard:
            if guard.tripped.is_set():
                break
            wait_cool(guard, a.pause_at, a.resume_at)
            if guard.tripped.is_set():
                break
        t0 = time.monotonic()
        try:
            result = run_one(server, a.token, page, force=False)
            with Image.open(page) as im:
                drawn = draw_page(im, result, font_at, floor)
            dst.parent.mkdir(parents=True, exist_ok=True)
            save(drawn, dst, a.quality)
        except Exception as e:  # noqa: BLE001 - report the page and move on
            failures.append(str(rel))
            print(f"[{i}/{len(pages)}] {rel}  FAILED {e}", file=sys.stderr, flush=True)
            continue
        index[str(rel)] = result
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1), "utf-8")
        temp = f"  junction {guard.junction:.0f}C" if guard and guard.junction is not None else ""
        print(f"[{i}/{len(pages)}] {rel}  {len(result)} regions  {time.monotonic() - t0:.1f}s{temp}",
              file=sys.stderr, flush=True)

    took = time.monotonic() - t_run
    if guard and guard.tripped.is_set():
        print(f"STOPPED by the MI50 guard ({guard.reason}); rerun to resume", file=sys.stderr)
        return 2
    peak = f", MI50 peak {guard.peak:.0f}C" if guard else ""
    print(f"done in {took:.0f}s{peak}; {len(failures)} failed {failures if failures else ''}",
          file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
