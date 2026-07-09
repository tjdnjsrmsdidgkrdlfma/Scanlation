"""Markdown report writers for the compare_models harness: the per-run OCR
timing+text tables, the batch speed summary, and the consolidated crop-OCR doc."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from compare.registry import all_adapters


def _write_ocr_report(out_path, crops: list[Image.Image], order: list[str],
                      texts: dict[str, list[str]], timings: dict[str, dict[str, float]],
                      *, quiet: bool = False) -> None:
    """Two markdown tables: a timing summary (engine x device: total + ms/crop, warm)
    then the transcribed text (one row per crop, one column per engine). quiet skips
    the stdout echo (used by the batch, which writes one report per image)."""
    engines = [e for e in order if e in texts]  # ran + produced text at least once
    dev_cols = [d for d in ("cpu", "cuda", "server")
                if any(d in timings.get(e, {}) for e in engines)]
    n = max(1, len(crops))

    L = [f"## OCR timing — {len(crops)} crops, warm (excludes first-call init)\n",
         "| engine | " + " | ".join(dev_cols) + " |",
         "|---|" + "|".join("---" for _ in dev_cols) + "|"]
    for e in engines:
        cells = []
        for d in dev_cols:
            ms = timings.get(e, {}).get(d)
            cells.append(f"{ms:.0f}ms ({ms / n:.1f}/crop)" if ms is not None else "—")
        L.append(f"| {e} | " + " | ".join(cells) + " |")
    if "cpu" in dev_cols and "cuda" in dev_cols:  # explicit speedup line per engine
        L.append("")
        for e in engines:
            c, g = timings.get(e, {}).get("cpu"), timings.get(e, {}).get("cuda")
            if c and g:
                L.append(f"- **{e}**: cuda {c / g:.1f}x faster than cpu ({c:.0f}ms -> {g:.0f}ms)")

    L += ["", "## OCR text — crops from ref detector\n",
          "| # | " + " | ".join(engines) + " |",
          "|---|" + "|".join("---" for _ in engines) + "|"]
    for i in range(len(crops)):
        cells = [(texts[e][i] if i < len(texts[e]) else "").replace("|", "\\|").replace("\n", " ") for e in engines]
        L.append(f"| {i:02d} | " + " | ".join(cells) + " |")

    report = "\n".join(L) + "\n"
    Path(out_path).write_text(report, encoding="utf-8")
    if not quiet:
        print("\n" + report)


def _write_ocr_summary(out_path: Path, n_images: int, total_crops: int,
                       order: list[str], agg: dict[str, dict[str, list]]) -> None:
    """Top-level CPU-vs-GPU speed table for the batch: total + ms/crop per engine
    and device, summed over every image (agg[engine][device] = [sum_ms, n_crops])."""
    engines = [e for e in order if e in agg]
    dev_cols = [d for d in ("cpu", "cuda", "server") if any(d in agg.get(e, {}) for e in engines)]
    L = [f"# OCR comparison — {n_images} images, {total_crops} crops (crops from the box model)\n",
         "Total OCR time per engine/device, warm (excludes first-call init).\n",
         "| engine | " + " | ".join(dev_cols) + " |",
         "|---|" + "|".join("---" for _ in dev_cols) + "|"]
    for e in engines:
        cells = []
        for d in dev_cols:
            v = agg.get(e, {}).get(d)
            cells.append(f"{v[0]:.0f}ms ({v[0] / max(1, v[1]):.1f}/crop)" if v else "—")
        L.append(f"| {e} | " + " | ".join(cells) + " |")
    if "cpu" in dev_cols and "cuda" in dev_cols:
        L.append("")
        for e in engines:
            c, g = agg.get(e, {}).get("cpu"), agg.get(e, {}).get("cuda")
            if c and g and g[0]:
                L.append(f"- **{e}**: cuda {c[0] / g[0]:.1f}x faster than cpu")
    report = "\n".join(L) + "\n"
    Path(out_path).write_text(report, encoding="utf-8")
    print("\n" + report)


def _consolidate_images(out_root: Path):
    """(rel, engine-cols, rows) per image from per-image ocr.json under out_root;
    rows[i] = each engine's text for crop i (aligned to the shared rtdetr crops)."""
    import json
    canon = [a.id for a in all_adapters() if a.kind == "ocr"]  # stable column order
    images = []
    for jf in sorted(out_root.rglob("ocr.json")):
        eng = json.loads(jf.read_text(encoding="utf-8")).get("engines", {})
        cols = [e for e in canon if e in eng]
        if not cols:
            continue
        n = max(len(eng[e]["texts"]) for e in cols)
        rows = [[(eng[e]["texts"][i] if i < len(eng[e]["texts"]) else "") for e in cols] for i in range(n)]
        images.append((jf.parent.relative_to(out_root).as_posix(), cols, rows))
    return images


def _write_ocr_md(dest: Path, images) -> None:
    esc = lambda s: (s or "").replace("|", "\\|").replace("\n", " ")  # noqa: E731
    L = [f"# OCR text comparison — {len(images)} images (crop recognizers, rows = rtdetr crops)\n"]
    for rel, cols, rows in images:
        L.append(f"\n## {rel}\n")
        L.append("| # | " + " | ".join(cols) + " |")
        L.append("|---|" + "|".join("---" for _ in cols) + "|")
        for i, row in enumerate(rows):
            L.append(f"| {i:02d} | " + " | ".join(esc(t) for t in row) + " |")
    dest.write_text("\n".join(L) + "\n", encoding="utf-8")
