"""Compare candidate DETECTION and Japanese-OCR models on the same manga page.

This is a RESEARCH harness, deliberately separate from the plugin system: each
candidate model is wrapped in a small adapter that lazy-imports its framework and
is *skipped with a clear reason* if its deps/weights aren't installed. So you can
enable models one at a time (pip install ..., pull an ollama tag) and re-run.

  * detect : every available detector draws its boxes on the page; the panels are
             tiled into ONE side-by-side montage (boxes + count + ms per model).
  * ocr    : one reference detector makes the deskewed crops, then every available
             OCR model reads the SAME crops; the text is printed aligned per crop.

    ../../venv/Scripts/python tools/compare_models.py list
    ../../venv/Scripts/python tools/compare_models.py detect page.png [--only ogkalu_rtdetr,kiuyha_yolo]   # -> compare_out/
    ../../venv/Scripts/python tools/compare_models.py ocr page.png [--ref-detector ogkalu_rtdetr] [--device both] [--only mangaocr,qwen3vl]

Once a winner is clear, promote just that model to a real scanlation-<name>
plugin (EngineBase + entry_points). This script stays throwaway research tooling,
so heavy frameworks (ultralytics, transformers VLMs) are NOT project deps — they
import lazily and only when their adapter actually runs.
"""
from __future__ import annotations

import argparse

from compare.commands import (
    cmd_list, cmd_detect, cmd_ocr, cmd_ocrbatch, cmd_consolidate, cmd_boxhtml,
    cmd_batch, cmd_ba,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show every adapter + whether it can run here").set_defaults(fn=cmd_list)

    d = sub.add_parser("detect", help="run every available detector -> side-by-side montage")
    d.add_argument("image")
    d.add_argument("--out", default="compare_out/compare_detectors.png")
    d.add_argument("--only", default=None, help="comma ids, e.g. ogkalu_rtdetr,ogkalu_yolov8m")
    d.add_argument("--exclude", default=None, help="comma ids to drop, e.g. kitsumed_seg")
    d.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                   help="tune an adapter attr, e.g. --opt conf=0.5 --opt nms_iou=0.5 (repeatable)")
    d.add_argument("--panels", default="compare_out/compare_panels", help="dir for full-res per-detector images")
    d.add_argument("--color-by-class", dest="by_class", action="store_true",
                   help="color boxes per class (default: all red, like the sample)")
    d.set_defaults(fn=cmd_detect)

    o = sub.add_parser("ocr", help="run every available OCR on the ref detector's crops")
    o.add_argument("image")
    o.add_argument("--ref-detector", default="ogkalu_rtdetr",
                   help="detector that makes the crops (the decided box model)")
    o.add_argument("--device", default=None, choices=["cpu", "cuda", "both"],
                   help="run OCR on cpu, cuda, or both (side-by-side timing); default: auto")
    o.add_argument("--only", default=None, help="comma ids, e.g. mangaocr,qwen3vl")
    o.add_argument("--exclude", default=None, help="comma ids to drop")
    o.add_argument("--max-crops", type=int, default=20)
    o.add_argument("--out", default="compare_out/compare_ocr.md")
    o.add_argument("--crops", default="compare_out/crops_ocr")
    o.set_defaults(fn=cmd_ocr)

    ob = sub.add_parser("ocrbatch", help="run OCR over a folder tree -> <out>/<category>/<image>/ocr.md")
    ob.add_argument("root", nargs="?", default="samples", help="input root (default: samples)")
    ob.add_argument("--out", default="compare_out", help="output root (mirrors the input tree)")
    ob.add_argument("--ref-detector", default="ogkalu_rtdetr", help="detector that makes the crops (box model)")
    ob.add_argument("--device", default=None, choices=["cpu", "cuda", "both"],
                    help="run OCR on cpu, cuda, or both (side-by-side timing); default: auto")
    ob.add_argument("--only", default=None, help="comma ids, e.g. mangaocr,paddleocr_vl")
    ob.add_argument("--exclude", default=None, help="comma ids to drop")
    ob.add_argument("--max-crops", type=int, default=20)
    ob.set_defaults(fn=cmd_ocrbatch)

    cs = sub.add_parser("consolidate", help="gather per-image ocr.json into one crop-OCR comparison doc (md/html)")
    cs.add_argument("--out", default="compare_out", help="tree with per-image ocr.json (default: compare_out)")
    cs.add_argument("--name", default="_compare_crops", help="output file stem under --out")
    cs.add_argument("--format", default="both", choices=["md", "html", "both"], help="default: both")
    cs.add_argument("--ref", default="mangaocr", help="engine others are diff-highlighted against in html")
    cs.add_argument("--link", action="store_true",
                    help="html: reference crop images by relative path instead of base64-embedding (smaller file)")
    cs.set_defaults(fn=cmd_consolidate)

    bh = sub.add_parser("boxhtml", help="gather detector <model>.png overlays into one BOX-scoring HTML (click-to-vote)")
    bh.add_argument("--out", default="compare_out", help="tree with per-image <model>.png overlays (default: compare_out)")
    bh.add_argument("--name", default="_compare_box", help="output file stem under --out")
    bh.add_argument("--embed", action="store_true",
                    help="base64-embed overlays (default: link by relative path — full-page PNGs are large)")
    bh.set_defaults(fn=cmd_boxhtml)

    b = sub.add_parser("batch", help="run detectors over a folder tree -> <out>/<category>/<image>/<model>.png")
    b.add_argument("root", nargs="?", default="samples", help="input root of category folders (default: samples)")
    b.add_argument("--out", default="compare_out", help="output root (mirrors the input tree)")
    b.add_argument("--only", default=None, help="comma ids, e.g. ogkalu_rtdetr,kitsumed_seg")
    b.add_argument("--exclude", default=None, help="comma ids to drop, e.g. kitsumed_seg")
    b.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                   help="tune an adapter attr, e.g. --opt conf=0.5 --opt nms_iou=0.5 (repeatable)")
    b.add_argument("--color-by-class", dest="by_class", action="store_true",
                   help="color boxes per class (default: all red)")
    b.set_defaults(fn=cmd_batch)

    ba = sub.add_parser("ba", help="before/after dedup for one detector -> <out>/<cat>/<img>/{before,after}.png")
    ba.add_argument("root", nargs="?", default="samples", help="input root (default: samples)")
    ba.add_argument("--out", default="compare_out", help="output root (mirrors the input tree)")
    ba.add_argument("--detector", default="ogkalu_rtdetr", help="detector to A/B on dedup")
    ba.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                    help="override the AFTER dedup, e.g. --opt nms_iou=0.5 --opt contain_thresh=0.7")
    ba.add_argument("--color-by-class", dest="by_class", action="store_true",
                    help="color boxes per class (default: all red)")
    ba.set_defaults(fn=cmd_ba)

    args = ap.parse_args()
    args.fn(args)
