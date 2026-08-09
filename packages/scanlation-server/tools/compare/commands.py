"""The compare_models subcommands (list/detect/ocr/ocrbatch/consolidate/boxhtml/
batch/ba/translate/translatehtml) and their shared helpers."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from PIL import Image

from compare.core import Adapter, DetResult, _resolve_device, _torch_device, dedup_boxes
from compare.registry import all_adapters, _select
from compare.render import render_panel, montage
from compare.report import (
    _write_ocr_report, _write_ocr_summary, _consolidate_images, _write_ocr_md,
    _write_translate_summary,
)
from compare.html import (
    _write_ocr_html, _consolidate_box_images, _write_box_html,
    _consolidate_translate_pages, _write_translate_html,
)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def _explicit_ids(only: str | None) -> set[str]:
    return {s.strip() for s in only.split(",")} if only else set()


def _coerce(cur, v: str):
    """Coerce a --opt string value to the type of the attribute it overrides."""
    if isinstance(cur, bool):
        return v.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int):
        return int(v)
    if isinstance(cur, float):
        return float(v)
    if cur is None:  # an off-by-default knob (e.g. nms_iou): accept a number
        try:
            return float(v)
        except ValueError:
            return v
    return v


def _apply_opts(adapters: list[Adapter], opts: list[str] | None) -> None:
    """Set tunable attributes on adapters from KEY=VALUE strings (e.g.
    conf=0.5 nms_iou=0.5). An adapter without the attribute silently ignores it,
    so `--opt conf=0.5` only affects the detectors that actually have a conf."""
    pairs = {}
    for o in opts or []:
        if "=" not in o:
            sys.exit(f"--opt expects KEY=VALUE, got {o!r}")
        k, v = o.split("=", 1)
        pairs[k.strip()] = v.strip()
    for a in adapters:
        for k, v in pairs.items():
            if hasattr(a, k):
                setattr(a, k, _coerce(getattr(a, k), v))


def _run_available(adapters: list[Adapter], explicit: set[str]):
    """Yield (adapter, ok, reason). Prints a live skip/run line per adapter. A
    `heavy` adapter is skipped unless the user named it in --only, so a bare run
    never kicks off a multi-GB model download by surprise."""
    for a in adapters:
        if a.heavy and a.id not in explicit:
            ok, why = False, "heavy (multi-GB) — add to --only to run"
        else:
            ok, why = a.available()
        status = "run " if ok else "SKIP"
        print(f"  [{status}] {a.id:<14} {a.label}" + ("" if ok else f"   -- {why}"), file=sys.stderr)
        yield a, ok, why


def cmd_list(_args) -> None:
    print("adapters (available on this machine = [run]):\n", file=sys.stderr)
    for a in all_adapters():
        ok, why = a.available()
        tag = "run " if ok else "SKIP"
        heavy = "  (heavy: name it in --only)" if a.heavy else ""
        print(f"  [{tag}] {a.kind:<7} {a.id:<14} {a.label}{heavy}")
        if not ok:
            print(f"           -> {why}\n           install: {a.install_hint}")


def cmd_detect(args) -> None:
    img = Image.open(args.image).convert("RGB")
    panel_dir = Path(args.panels)
    panel_dir.mkdir(parents=True, exist_ok=True)
    ads = _select("detect", args.only, args.exclude)
    _apply_opts(ads, args.opt)
    panels: list[Image.Image] = []
    for a, ok, _ in _run_available(ads, _explicit_ids(args.only)):
        if not ok:
            continue
        try:
            a.load()
            t0 = time.perf_counter()
            boxes = a.detect(img)
            res = DetResult(boxes, (time.perf_counter() - t0) * 1000)
            print(f"    {a.id}: {len(boxes)} boxes in {res.ms:.0f}ms", file=sys.stderr)
            panel = render_panel(img, res, a.label, by_class=args.by_class)
            panel.save(panel_dir / f"{a.id}.png")  # full-res, for zooming in
            panels.append(panel)
        except Exception as exc:  # noqa: BLE001 - one broken model must not kill the run
            print(f"    {a.id}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    montage(panels, out)
    print(f"\nwrote {out}  +  {len(panels)} full-res panel(s) -> {panel_dir}/", file=sys.stderr)


def _load_ref_detector(ref_id: str) -> Adapter:
    """Load the reference detector (the decided box model) once, so a batch can
    reuse it across every image instead of reloading it per image."""
    ad = next((a for a in all_adapters() if a.id == ref_id and a.kind == "detect"), None)
    if ad is None:
        sys.exit(f"unknown ref detector: {ref_id}")
    ok, why = ad.available()
    if not ok:
        sys.exit(f"ref detector '{ref_id}' unavailable: {why}")
    ad.load()
    return ad


def _crops_from_boxes(img: Image.Image, ref: Adapter, max_crops: int) -> list[Image.Image]:
    """Deskewed crops from an already-loaded ref detector's boxes on one page."""
    from app.geometry import deskew_crop
    from scanlation_sdk.contracts import Region
    crops: list[Image.Image] = []
    for b in ref.detect(img)[:max_crops]:
        if b.polygon and len(b.polygon) == 4:  # rotated quad -> deskew upright
            import numpy as np
            quad = np.array(b.polygon, dtype=np.float32)
            crops.append(deskew_crop(img, Region.from_quad(quad, angle=0.0, vertical=False)))
        else:
            crops.append(img.crop(tuple(int(v) for v in b.xyxy)))
    return crops


def _crops_from_ref(img: Image.Image, ref_id: str, max_crops: int) -> list[Image.Image]:
    """The SAME crops every OCR model reads, from the reference detector (default
    the decided box model) — so the comparison is apples-to-apples."""
    return _crops_from_boxes(img, _load_ref_detector(ref_id), max_crops)


def _ocr_devices(choice: str | None) -> list[str | None]:
    """Devices each OCR engine is run on. 'both' = CPU and (if present) CUDA in one
    run so the speed gap is side by side; explicit 'cpu'/'cuda' forces one; None =
    auto (best available), the old behaviour. Falls back to CPU with a note when
    CUDA is asked for but absent."""
    if choice == "both":
        devs: list[str | None] = ["cpu"]
        if _torch_device() == "cuda":
            devs.append("cuda")
        else:
            print("note: no CUDA device — 'both' shows CPU only", file=sys.stderr)
        return devs
    if choice == "cuda" and _torch_device() != "cuda":
        print("note: no CUDA device — running on CPU instead", file=sys.stderr)
        return ["cpu"]
    if choice in ("cpu", "cuda"):
        return [choice]
    return [None]  # auto: whatever the engine picks


def _run_label(adapter: Adapter, dev: str | None) -> str:
    """Column name for one (engine, device) run: the resolved device for engines we
    drive, or 'server' for ollama (its CPU/GPU choice is the server's business)."""
    return _resolve_device(dev) if adapter.device_switchable else "server"


def _run_devices(adapter: Adapter, devices: list[str | None]) -> list[str | None]:
    """The device list to run one OCR adapter over: `devices` for engines we drive,
    [None] for ollama (its server picks), cpu dropped for cpu_ok=False models."""
    run_devs = devices if adapter.device_switchable else [None]  # ollama: server picks the device
    return [dv for dv in run_devs if adapter.cpu_ok or dv != "cpu"]  # skip cpu for cpu_ok=False (e.g. 7B)


def _time_recognize(adapter: Adapter, crops: list[Image.Image]) -> tuple[list[str], float]:
    """OCR every crop, returning (texts, total_ms). One warm-up call first so the
    timing excludes one-off init (CUDA kernels / lazy weights) and CPU-vs-GPU is fair."""
    if crops:
        adapter.recognize(crops[0])  # warm up, untimed
    t0 = time.perf_counter()
    texts = [adapter.recognize(c) for c in crops]
    return texts, (time.perf_counter() - t0) * 1000


def cmd_ocr(args) -> None:
    img = Image.open(args.image).convert("RGB")
    crops = _crops_from_ref(img, args.ref_detector, args.max_crops)
    print(f"\n{len(crops)} crops from ref detector '{args.ref_detector}'", file=sys.stderr)

    crop_dir = Path(args.crops)
    crop_dir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(crops):
        c.save(crop_dir / f"crop_{i:02d}.png")

    devices = _ocr_devices(args.device)
    print(f"OCR on device(s): {', '.join(str(d or 'auto') for d in devices)}"
          f"   (torch cuda: {'yes' if _torch_device() == 'cuda' else 'no'})", file=sys.stderr)

    order: list[str] = []                       # engine ids, run order (table column order)
    texts: dict[str, list[str]] = {}            # id -> per-crop text (device-independent)
    timings: dict[str, dict[str, float]] = {}   # id -> {device_label: total_ms}
    for a, ok, _ in _run_available(_select("ocr", args.only, args.exclude), _explicit_ids(args.only)):
        if not ok:
            continue
        run_devs = _run_devices(a, devices)
        if not run_devs:
            print(f"    {a.id}: skipped — needs a GPU but this run has no cuda device", file=sys.stderr)
            continue
        order.append(a.id)
        for dev in run_devs:
            a.device = dev
            lab = _run_label(a, dev)
            try:
                a.load()  # re-load per device (that IS the CPU-vs-GPU comparison)
                out, ms = _time_recognize(a, crops)
                texts.setdefault(a.id, out)  # text is device-independent; keep the first run's
                timings.setdefault(a.id, {})[lab] = ms
                print(f"    {a.id}[{lab}]: {ms:.0f}ms ({ms / max(1, len(crops)):.1f}ms/crop)", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 - one broken engine/device must not kill the run
                print(f"    {a.id}[{lab}]: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _write_ocr_report(args.out, crops, order, texts, timings)
    print(f"\nwrote {args.out}  +  {len(crops)} crops -> {crop_dir}/", file=sys.stderr)


def cmd_ocrbatch(args) -> None:
    """OCR every image under a folder tree, model-outer (each engine loads once per
    device and is reused across all images). Crops come from the ref detector (the
    decided box model). Output mirrors the input tree, taking the detection outputs'
    old place:  <out>/<category>/<image>/{ocr.md, ocr.json} (+ crops/), plus a
    top-level _ocr_summary.md with the CPU-vs-GPU speed table over all images.

    Results ACCUMULATE: each run merges the engines it produced into the per-image
    ocr.json, so a new engine can be added later (a fresh dep, a heavy GPU-only
    model, ollama) with `--only that_engine` without re-running — or losing — the
    others. (Prior results are reused only when the ref detector and crop count match.)"""
    import json
    root, out_root = Path(args.root), Path(args.out)
    tasks = _collect_tasks(root)
    if not tasks:
        sys.exit(f"no images found under {root}")
    devices = _ocr_devices(args.device)
    print(f"{len(tasks)} images under {root}; ref='{args.ref_detector}'; "
          f"OCR device(s): {', '.join(str(d or 'auto') for d in devices)}\n", file=sys.stderr)

    # 1) ref detector loads once; crops per image saved + cached. Any prior ocr.json
    #    (same ref + crop count) is loaded so its engine columns survive this run.
    ref = _load_ref_detector(args.ref_detector)
    crops_by_img: dict[tuple[str, str], list[Image.Image]] = {}
    prev: dict[tuple[str, str], dict] = {}
    for cat, img_path in tasks:
        key = (cat, img_path.stem)
        crops = _crops_from_boxes(Image.open(img_path).convert("RGB"), ref, args.max_crops)
        crops_by_img[key] = crops
        d = out_root / cat / img_path.stem
        (d / "crops").mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(crops):
            c.save(d / "crops" / f"crop_{i:02d}.png")
        jf = d / "ocr.json"
        if jf.exists():
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                if data.get("ref") == args.ref_detector and data.get("n_crops") == len(crops):
                    prev[key] = data.get("engines", {})
                else:
                    print(f"  note: {cat}/{img_path.stem}: prior ocr.json dropped (ref/crop-count changed)", file=sys.stderr)
            except Exception:  # noqa: BLE001
                pass
    total_crops = sum(len(v) for v in crops_by_img.values())
    print(f"ref '{args.ref_detector}': {total_crops} crops over {len(tasks)} images\n", file=sys.stderr)

    # 2) each engine x device, model-outer: load once, then read every image's crops
    this_run: dict[tuple[str, str], dict[str, dict]] = {}  # img -> engine -> {"texts":[...], "timings":{dev:ms}}
    for a, ok, _ in _run_available(_select("ocr", args.only, args.exclude), _explicit_ids(args.only)):
        if not ok:
            continue
        run_devs = _run_devices(a, devices)
        if not run_devs:
            print(f"  {a.id}: skipped — needs a GPU (cpu_ok=False) but this run has no cuda device", file=sys.stderr)
            continue
        for dev in run_devs:
            a.device = dev
            lab = _run_label(a, dev)
            try:
                a.load()
            except Exception as exc:  # noqa: BLE001
                print(f"  {a.id}[{lab}]: LOAD ERROR {type(exc).__name__}: {exc} — skipping", file=sys.stderr)
                continue
            warmed, run_row = False, [0.0, 0]
            for cat, img_path in tasks:
                key = (cat, img_path.stem)
                crops = crops_by_img[key]
                if not crops:
                    continue
                try:
                    if not warmed:
                        a.recognize(crops[0])  # one warm-up per (engine, device), untimed
                        warmed = True
                    t0 = time.perf_counter()
                    out = [a.recognize(c) for c in crops]
                    ms = (time.perf_counter() - t0) * 1000
                    e = this_run.setdefault(key, {}).setdefault(a.id, {"texts": None, "timings": {}})
                    if e["texts"] is None:
                        e["texts"] = out  # text is device-independent; keep the first device's
                    e["timings"][lab] = ms
                    run_row[0] += ms
                    run_row[1] += len(crops)
                except Exception as exc:  # noqa: BLE001 - one broken image must not kill the run
                    print(f"    {cat}/{img_path.stem} {a.id}[{lab}]: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            if run_row[1]:
                print(f"  {a.id}[{lab}]: {run_row[0]:.0f}ms over {run_row[1]} crops "
                      f"({run_row[0] / run_row[1]:.1f}ms/crop)", file=sys.stderr)

    # 3) merge prior + this run per image; rewrite ocr.md + ocr.json; feed the summary
    canon = [a.id for a in all_adapters() if a.kind == "ocr"]  # stable column order
    global_agg: dict[str, dict[str, list]] = {}
    for cat, img_path in tasks:
        key = (cat, img_path.stem)
        merged = {**prev.get(key, {}), **this_run.get(key, {})}  # this run's engines win
        if not merged:
            continue
        order = [e for e in canon if e in merged]
        texts = {e: merged[e]["texts"] for e in order}
        timings = {e: merged[e]["timings"] for e in order}
        d = out_root / cat / img_path.stem
        _write_ocr_report(d / "ocr.md", crops_by_img[key], order, texts, timings, quiet=True)
        (d / "ocr.json").write_text(json.dumps(
            {"ref": args.ref_detector, "n_crops": len(crops_by_img[key]), "engines": merged},
            ensure_ascii=False, indent=1), encoding="utf-8")
        for e in order:  # accumulate the summary over all images
            for dev, ms in timings[e].items():
                row = global_agg.setdefault(e, {}).setdefault(dev, [0.0, 0])
                row[0] += ms
                row[1] += len(crops_by_img[key])

    _write_ocr_summary(out_root / "_ocr_summary.md", len(tasks), total_crops,
                       [e for e in canon if e in global_agg], global_agg)
    print(f"\nwrote {out_root}/<category>/<image>/{{ocr.md, ocr.json}}  +  {out_root}/_ocr_summary.md", file=sys.stderr)


def _collect_tasks(root: Path) -> list[tuple[str, Path]]:
    """(category, image_path) for every image under root, category = its subfolder
    (so samples/노이즈/x.jpg -> ('노이즈', .../x.jpg)). Loose images -> '_root'."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    tasks = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            rel = p.relative_to(root).parent
            tasks.append((str(rel).replace("\\", "/") if str(rel) != "." else "_root", p))
    return tasks


def cmd_batch(args) -> None:
    """Run every available detector over a folder tree, model-outer so each model
    loads once and is reused across all images. Output mirrors the input tree:
    <out>/<category>/<image_stem>/<model>.png  (+ a _montage.png per image)."""
    root, out_root = Path(args.root), Path(args.out)
    tasks = _collect_tasks(root)
    if not tasks:
        sys.exit(f"no images found under {root}")
    print(f"{len(tasks)} images under {root} -> {out_root}/<category>/<image>/<model>.png\n", file=sys.stderr)

    ads = _select("detect", args.only, args.exclude)
    _apply_opts(ads, args.opt)
    adapters = [a for a, ok, _ in _run_available(ads, _explicit_ids(args.only)) if ok]
    if not adapters:
        sys.exit("no detectors available (run `list`)")

    counts: dict[tuple[str, str], dict[str, int]] = {}
    for a in adapters:
        try:
            a.load()
        except Exception as exc:  # noqa: BLE001
            print(f"  {a.id}: LOAD ERROR {type(exc).__name__}: {exc} — skipping", file=sys.stderr)
            continue
        print(f"  {a.id}: running on {len(tasks)} images ...", file=sys.stderr)
        for cat, img_path in tasks:
            try:
                img = Image.open(img_path).convert("RGB")
                t0 = time.perf_counter()
                boxes = a.detect(img)
                ms = (time.perf_counter() - t0) * 1000
                panel = render_panel(img, DetResult(boxes, ms), a.label, by_class=args.by_class)
                d = out_root / cat / img_path.stem
                d.mkdir(parents=True, exist_ok=True)
                panel.save(d / f"{a.id}.png")
                counts.setdefault((cat, img_path.stem), {})[a.id] = len(boxes)
            except Exception as exc:  # noqa: BLE001
                print(f"    {cat}/{img_path.name} {a.id}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)

    for cat, img_path in tasks:  # a side-by-side montage per image, from the saved panels
        d = out_root / cat / img_path.stem
        panels = [Image.open(d / f"{a.id}.png") for a in adapters if (d / f"{a.id}.png").exists()]
        if panels:
            montage(panels, d / "_montage.png")

    ids = [a.id for a in adapters]
    print("\n=== box counts per image (category / image) ===")
    last = None
    for (cat, stem), c in sorted(counts.items()):
        if cat != last:
            print(f"\n[{cat}]")
            last = cat
        print(f"  {stem}: " + "  ".join(f"{i}={c.get(i, '-')}" for i in ids))
    print(f"\nwrote {out_root}/  ({len(tasks)} images x {len(adapters)} models)", file=sys.stderr)


def cmd_ba(args) -> None:
    """Before/after for ONE detector's dedup: per image render it with dedup OFF
    (before.png) and ON (after.png) into <out>/<category>/<image>/, keeping the
    input tree. 'After' reuses 'before's boxes deduped (one inference per image);
    the after thresholds are the adapter's configured nms_iou/contain_thresh,
    overridable with --opt (e.g. --opt nms_iou=0.5)."""
    root, out_root = Path(args.root), Path(args.out)
    tasks = _collect_tasks(root)
    if not tasks:
        sys.exit(f"no images found under {root}")
    ads = _select("detect", args.detector, None)
    _apply_opts(ads, args.opt)
    ad = next((a for a in ads if a.id == args.detector), None)
    if ad is None:
        sys.exit(f"unknown detector: {args.detector} (run `list`)")
    ok, why = ad.available()
    if not ok:
        sys.exit(f"{args.detector} unavailable: {why}")
    ad.load()
    after_iou, after_contain = getattr(ad, "nms_iou", None), getattr(ad, "contain_thresh", None)
    if after_iou is None and after_contain is None:
        print(f"note: {args.detector} has no dedup configured — after == before", file=sys.stderr)
    conf = getattr(ad, "conf", "?")
    before_title = f"BEFORE  {args.detector}  conf={conf}  dedup=off"
    after_title = f"AFTER  {args.detector}  conf={conf}  nms_iou={after_iou}  contain={after_contain}"
    print(f"{len(tasks)} images: {args.detector}  before(off) vs after(nms_iou={after_iou}, contain_thresh={after_contain})\n",
          file=sys.stderr)
    for cat, img_path in tasks:
        img = Image.open(img_path).convert("RGB")
        if hasattr(ad, "nms_iou"):  # force raw for 'before'
            ad.nms_iou, ad.contain_thresh = None, None
        before = ad.detect(img)
        after = dedup_boxes(before, after_iou, after_contain)
        d = out_root / cat / img_path.stem
        d.mkdir(parents=True, exist_ok=True)
        render_panel(img, DetResult(before, 0.0), before_title, by_class=args.by_class).save(d / "before.png")
        render_panel(img, DetResult(after, 0.0), after_title, by_class=args.by_class).save(d / "after.png")
        print(f"  {cat}/{img_path.stem}: {len(before)} -> {len(after)}")
    print(f"\nwrote {out_root}/<category>/<image>/{{before,after}}.png  ({len(tasks)} images)", file=sys.stderr)


def cmd_consolidate(args) -> None:
    """Gather every image's crop-OCR into ONE comparison doc (md and/or html). Rows =
    the shared rtdetr crops, columns = engines, from per-image ocr.json under <out>.
    The html char-diffs each engine against --ref (shared text green, differences red)
    so agreement vs divergence is visible at a glance. (Page-level engines aren't in
    ocr.json, so only crop recognizers appear.)"""
    out_root = Path(args.out)
    images = _consolidate_images(out_root)
    if not images:
        sys.exit(f"no usable ocr.json under {out_root} (run ocrbatch first)")
    made = []
    if args.format in ("md", "both"):
        _write_ocr_md(out_root / f"{args.name}.md", images)
        made.append(f"{args.name}.md")
    if args.format in ("html", "both"):
        _write_ocr_html(out_root / f"{args.name}.html", images, args.ref, out_root, embed=not args.link)
        made.append(f"{args.name}.html")
    print(f"wrote {out_root}/{{{', '.join(made)}}}  ({len(images)} images)", file=sys.stderr)


# --------------------------------------------------------------------------- #
# translate
# --------------------------------------------------------------------------- #
class _FallbackCounter(logging.Handler):
    """Counts the translator's "batch of N failed" warnings while one model runs.

    That failure is a quality signal in its own right, not just plumbing: it means the
    model broke the batch JSON — usually by locking into a repetition loop on an SFX —
    and the page had to be redone one text at a time. A model that needs it often is
    worse in production even if its prose reads well. Every warning is echoed too,
    since attaching any handler is what stops logging's last-resort one from printing."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.n = 0

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "batch of" in msg:
            self.n += 1
        print(f"      {msg}", file=sys.stderr)


def _cool_gate(cmd: str) -> None:
    """Block on an external readiness command before sending more work.

    A blind pause cannot pace a backend whose limit is thermal: too short and the
    card still climbs, too long and the run crawls. The one thing that actually
    knows when to continue is the machine holding the card, so let it decide --
    the command blocks while the GPU is too hot and returns when it is not.

    Deliberately opaque: the harness must not learn how to reach the GPU host.
    Failures are ignored so a broken probe degrades to no pacing rather than
    stopping a run mid-corpus."""
    if not cmd:
        return
    import subprocess
    try:
        subprocess.run(cmd, shell=True, timeout=900, capture_output=True)
    except Exception as exc:  # noqa: BLE001
        print(f"      cool-cmd failed ({type(exc).__name__}: {exc}) — continuing unpaced",
              file=sys.stderr)


def _chunk_texts(texts: list[str], max_chars: int) -> list[list[int]]:
    """Group a page's text indices into sub-batches of at most max_chars of source.

    One call per page is the production shape, but it is also one long uninterrupted
    generation -- on a backend that can be cut off mid-request (a temperature stop,
    a restart) a page that takes 20s never lands, and the pass makes no progress at
    all. Splitting bounds how much work a single interruption can throw away.

    A single text is never split: these are speech bubbles, and half a bubble asks
    the model to translate a fragment. 0 = off, one call per page."""
    if max_chars <= 0:
        return [list(range(len(texts)))]
    groups: list[list[int]] = []
    cur: list[int] = []
    n = 0
    for i, t in enumerate(texts):
        if cur and n + len(t) > max_chars:
            groups.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += len(t)
    if cur:
        groups.append(cur)
    return groups


def _prior_translations(pages, source_engine: str, src: str, dst: str) -> dict[str, dict]:
    """rel -> {model: result} already on disk, keeping only what still answers the
    same question (same source engine, language pair and text count). Read once, up
    front, so a pass can both SKIP pages it already has and merge onto them after."""
    import json
    prev: dict[str, dict] = {}
    for rel, d, texts in pages:
        jf = d / "translate.json"
        if not jf.exists():
            continue
        try:
            old = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if (old.get("source_engine") == source_engine and old.get("src") == src
                and old.get("dst") == dst and old.get("n") == len(texts)):
            prev[rel] = old.get("models", {})
        else:
            print(f"  note: {rel}: prior translate.json dropped (source/langs/count changed)",
                  file=sys.stderr)
    return prev


def _translate_pages(out_root, source_engine: str) -> list[tuple[str, object, list[str]]]:
    """(rel, dir, texts) per page, read from the recognizer batch's ocr.json.

    The source text is NOT re-recognized here: every translator reads the exact same
    input, so a recognizer slip is identical across models instead of adding noise to
    the comparison. Pages whose ocr.json never ran that engine are skipped aloud."""
    import json
    pages = []
    for jf in sorted(out_root.rglob("ocr.json")):
        rel = jf.parent.relative_to(out_root).as_posix()
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  note: {rel}: unreadable ocr.json ({exc}) — skipped", file=sys.stderr)
            continue
        eng = (data.get("engines") or {}).get(source_engine) or {}
        texts = eng.get("texts")
        if not texts:
            print(f"  note: {rel}: no '{source_engine}' texts in ocr.json — skipped", file=sys.stderr)
            continue
        pages.append((rel, jf.parent, list(texts)))
    return pages


def cmd_translate(args) -> None:
    """Translate every page's recognized text with each available LLM, model-outer.

    A page goes to the model as ONE batch call — the production path — so the model
    sees a page's lines together and the run also exercises the batch JSON schema and
    its per-text fallback. Output lands beside the recognizer's as
    <out>/<category>/<image>/translate.json and, like ocrbatch, ACCUMULATES: because
    llama-server hosts one -hf at a time, each model is a separate pass (restart the
    server, re-run) and merges into what the earlier passes left."""
    import json
    import logging
    import os
    out_root = Path(args.out)
    # Bound per-request wait generously: the SDK default (10s) is tuned for a warm
    # production translator, and tripping it here would silently score the fallback
    # path instead of the model.
    os.environ["SCANLATION_HTTP_TIMEOUT"] = str(args.timeout)

    pages = _translate_pages(out_root, args.source_engine)
    if not pages:
        sys.exit(f"no pages with '{args.source_engine}' text under {out_root} (run ocrbatch first)")
    total_texts = sum(len(t) for _r, _d, t in pages)
    print(f"{len(pages)} pages / {total_texts} texts from '{args.source_engine}' "
          f"({args.src} -> {args.dst}); timeout {args.timeout:.0f}s\n", file=sys.stderr)

    prev_by_rel = _prior_translations(pages, args.source_engine, args.src, args.dst)

    this_run: dict[str, dict[str, dict]] = {}  # rel -> model -> {"texts", "ms", "fallbacks"}
    for a, ok, _ in _run_available(_select("translate", args.only, args.exclude), _explicit_ids(args.only)):
        if not ok:
            continue
        try:
            a.load()
        except Exception as exc:  # noqa: BLE001
            print(f"  {a.id}: LOAD ERROR {type(exc).__name__}: {exc} — skipping", file=sys.stderr)
            continue
        todo = [p for p in pages if not (args.resume and a.id in prev_by_rel.get(p[0], {}))]
        if args.resume and len(todo) < len(pages):
            print(f"  {a.id}: resuming — {len(pages) - len(todo)} page(s) already done, "
                  f"{len(todo)} to go", file=sys.stderr)
        if not todo:
            continue
        counter = _FallbackCounter()
        logging.getLogger("scanlation").addHandler(counter)
        run_ms, run_n = 0.0, 0
        try:
            for rel, _d, texts in todo:
                before = counter.n
                try:
                    t0 = time.perf_counter()
                    out = [""] * len(texts)
                    for grp in _chunk_texts(texts, args.max_chars):
                        _cool_gate(args.cool_cmd)
                        sub = a.translate_page([texts[i] for i in grp], args.src, args.dst)
                        for i, tr in zip(grp, sub):
                            out[i] = tr
                    ms = (time.perf_counter() - t0) * 1000
                except Exception as exc:  # noqa: BLE001 - one bad page must not kill the pass
                    print(f"    {rel} {a.id}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue
                this_run.setdefault(rel, {})[a.id] = {
                    "texts": out, "ms": ms, "fallbacks": counter.n - before,
                }
                run_ms += ms
                run_n += len(texts)
                # Rest in proportion to the work just done. The MI50 has no thermal
                # throttle, and a dense 27B+ decodes slowly enough to sit under load
                # for minutes and reach the junction cut mid-corpus -- which ends the
                # pass with a partial model. Scaling the pause to the page's own time
                # bounds the duty cycle without a per-model constant: a fast MoE page
                # barely pauses, a slow dense one rests as long as it ran.
                if args.cool_ratio > 0:
                    time.sleep(ms / 1000 * args.cool_ratio)
        finally:
            logging.getLogger("scanlation").removeHandler(counter)
        if run_n:
            print(f"  {a.id}: {run_ms / 1000:.1f}s over {len(todo)} pages / {run_n} texts "
                  f"({run_ms / len(todo):.0f}ms/page, {run_ms / run_n:.0f}ms/text)"
                  + (f", {counter.n} batch fallback(s)" if counter.n else ""), file=sys.stderr)

    # merge each page's prior models with this pass's, then rewrite translate.json.
    # Prior results survive only when they answer the same question (same source
    # engine, same language pair, same text count).
    canon = [a.id for a in all_adapters() if a.kind == "translate"]  # stable column order
    agg: dict[str, list] = {}  # model -> [sum_ms, n_texts, n_pages, fallbacks]
    for rel, d, texts in pages:
        jf = d / "translate.json"
        merged = {**prev_by_rel.get(rel, {}), **this_run.get(rel, {})}  # this pass's models win
        if not merged:
            continue
        jf.write_text(json.dumps({
            "source_engine": args.source_engine, "src": args.src, "dst": args.dst,
            "n": len(texts), "source": texts,
            "models": {m: merged[m] for m in canon if m in merged},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        for m, r in merged.items():
            row = agg.setdefault(m, [0.0, 0, 0, 0])
            row[0] += r.get("ms", 0.0)
            row[1] += len(texts)
            row[2] += 1
            row[3] += r.get("fallbacks", 0)

    _write_translate_summary(out_root / "_translate_summary.md", args.source_engine,
                             args.src, args.dst, [m for m in canon if m in agg], agg)
    print(f"\nwrote {out_root}/<category>/<image>/translate.json  +  {out_root}/_translate_summary.md",
          file=sys.stderr)


def cmd_translatehtml(args) -> None:
    """Gather every page's translate.json into ONE scoring HTML: rows are the shared
    crops (image + the recognized source), columns the candidate models, each cell
    clickable to vote 'this model translated this line best'. Sibling of consolidate
    (recognizers) and boxhtml (detectors), with its own vote namespace, so the three
    scorings never share tallies."""
    out_root = Path(args.out)
    pages = _consolidate_translate_pages(out_root)
    if not pages:
        sys.exit(f"no translate.json under {out_root} (run `translate` first)")
    dest = out_root / f"{args.name}.html"
    _write_translate_html(dest, pages, out_root, embed=not args.link)
    print(f"wrote {dest}  ({len(pages)} pages)", file=sys.stderr)


def cmd_boxhtml(args) -> None:
    """Gather detector box-overlays (compare_out/<cat>/<img>/<model>.png from `batch`) into
    ONE scoring HTML — per image each model's overlay side by side, click a panel to vote,
    per-model + per-category tallies (localStorage, boxsel: namespace). Detector analog of consolidate."""
    out_root = Path(args.out)
    images = _consolidate_box_images(out_root)
    if not images:
        sys.exit(f"no detector overlays under {out_root} (run `batch` first)")
    _write_box_html(out_root / f"{args.name}.html", images, out_root, embed=args.embed)
    print(f"wrote {out_root}/{args.name}.html  ({len(images)} images)", file=sys.stderr)
