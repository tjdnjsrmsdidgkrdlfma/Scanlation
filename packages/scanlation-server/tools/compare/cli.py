"""Compare candidate DETECTION, Japanese-OCR and TRANSLATOR models on the same pages.

This is a RESEARCH harness, deliberately separate from the plugin system: each
candidate model is wrapped in a small adapter that lazy-imports its framework and
is *skipped with a clear reason* if its deps/weights aren't installed. So you can
enable models one at a time (pip install ..., pull an ollama tag, restart
llama-server on the next -hf) and re-run.

  * detect    : every available detector draws its boxes on the page; the panels are
                tiled into ONE side-by-side montage (boxes + count + ms per model).
  * ocr       : one reference detector makes the deskewed crops, then every available
                OCR model reads the SAME crops; the text is printed aligned per crop.
  * translate : every available LLM translates the SAME recognized text (from the
                ocr batch's ocr.json), one batch call per page like the pipeline does.

    ../../venv/Scripts/python tools/compare_models.py list
    ../../venv/Scripts/python tools/compare_models.py detect page.png [--only ogkalu_rtdetr,kiuyha_yolo]   # -> compare_out/
    ../../venv/Scripts/python tools/compare_models.py ocr page.png [--ref-detector ogkalu_rtdetr] [--device both] [--only mangaocr,qwen3vl]
    ../../venv/Scripts/python tools/compare_models.py translate [--source-engine paddleocr_manga] [--only gemma-4-31B]
    ../../venv/Scripts/python tools/compare_models.py translatehtml   # -> compare_out/_compare_translate.html

Once a winner is clear, promote just that model to a real scanlation-<name>
plugin (EngineBase + entry_points). This script stays throwaway research tooling,
so heavy frameworks (ultralytics, transformers VLMs) are NOT project deps — they
import lazily and only when their adapter actually runs.
"""
from __future__ import annotations

import argparse

from compare.commands import (
    cmd_list, cmd_detect, cmd_ocr, cmd_ocrbatch, cmd_consolidate, cmd_boxhtml,
    cmd_batch, cmd_ba, cmd_translate, cmd_translatehtml,
)

DEFAULT_OUT = "compare_out"    # output root every subcommand writes under
DEFAULT_REF = "ogkalu_rtdetr"  # the decided detector model — makes the crops for the OCR commands
DEFAULT_SOURCE = "paddleocr_manga"  # the decided recognizer — its text is what gets translated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show every adapter + whether it can run here").set_defaults(fn=cmd_list)

    d = sub.add_parser("detect", help="run every available detector -> side-by-side montage")
    d.add_argument("image")
    d.add_argument("--out", default=f"{DEFAULT_OUT}/compare_detectors.png")
    d.add_argument("--only", default=None, help="comma ids, e.g. ogkalu_rtdetr,ogkalu_yolov8m")
    d.add_argument("--exclude", default=None, help="comma ids to drop, e.g. kitsumed_seg")
    d.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                   help="tune an adapter attr, e.g. --opt conf=0.5 --opt nms_iou=0.5 (repeatable)")
    d.add_argument("--panels", default=f"{DEFAULT_OUT}/compare_panels", help="dir for full-res per-detector images")
    d.add_argument("--color-by-class", dest="by_class", action="store_true",
                   help="color boxes per class (default: all red, like the sample)")
    d.set_defaults(fn=cmd_detect)

    o = sub.add_parser("ocr", help="run every available OCR on the ref detector's crops")
    o.add_argument("image")
    o.add_argument("--ref-detector", default=DEFAULT_REF,
                   help="detector that makes the crops (the decided box model)")
    o.add_argument("--device", default=None, choices=["cpu", "cuda", "both"],
                   help="run OCR on cpu, cuda, or both (side-by-side timing); default: auto")
    o.add_argument("--only", default=None, help="comma ids, e.g. mangaocr,qwen3vl")
    o.add_argument("--exclude", default=None, help="comma ids to drop")
    o.add_argument("--max-crops", type=int, default=20)
    o.add_argument("--out", default=f"{DEFAULT_OUT}/compare_ocr.md")
    o.add_argument("--crops", default=f"{DEFAULT_OUT}/crops_ocr")
    o.set_defaults(fn=cmd_ocr)

    ob = sub.add_parser("ocrbatch", help="run OCR over a folder tree -> <out>/<category>/<image>/ocr.md")
    ob.add_argument("root", nargs="?", default="samples", help="input root (default: samples)")
    ob.add_argument("--out", default=DEFAULT_OUT, help="output root (mirrors the input tree)")
    ob.add_argument("--ref-detector", default=DEFAULT_REF, help="detector that makes the crops (box model)")
    ob.add_argument("--device", default=None, choices=["cpu", "cuda", "both"],
                    help="run OCR on cpu, cuda, or both (side-by-side timing); default: auto")
    ob.add_argument("--only", default=None, help="comma ids, e.g. mangaocr,paddleocr_vl")
    ob.add_argument("--exclude", default=None, help="comma ids to drop")
    ob.add_argument("--max-crops", type=int, default=20)
    ob.set_defaults(fn=cmd_ocrbatch)

    cs = sub.add_parser("consolidate", help="gather per-image ocr.json into one crop-OCR comparison doc (md/html)")
    cs.add_argument("--out", default=DEFAULT_OUT, help="tree with per-image ocr.json (default: compare_out)")
    cs.add_argument("--name", default="_compare_crops", help="output file stem under --out")
    cs.add_argument("--format", default="both", choices=["md", "html", "both"], help="default: both")
    cs.add_argument("--ref", default="mangaocr", help="engine others are diff-highlighted against in html")
    cs.add_argument("--link", action="store_true",
                    help="html: reference crop images by relative path instead of base64-embedding (smaller file)")
    cs.set_defaults(fn=cmd_consolidate)

    tr = sub.add_parser("translate", help="translate every page's recognized text with each available LLM")
    tr.add_argument("--out", default=DEFAULT_OUT, help="tree with per-image ocr.json (default: compare_out)")
    tr.add_argument("--source-engine", dest="source_engine", default=DEFAULT_SOURCE,
                    help="recognizer whose ocr.json text every model translates (the decided OCR model)")
    tr.add_argument("--src", default="ja", help="source language (iso1)")
    tr.add_argument("--dst", default="ko", help="target language (iso1)")
    tr.add_argument("--only", default=None, help="comma ids, e.g. gemma-4-31B,Qwen3.6-27B")
    tr.add_argument("--exclude", default=None, help="comma ids to drop")
    tr.add_argument("--cool-cmd", dest="cool_cmd", default="",
                    help="shell command run before each request that should BLOCK while the "
                         "backend's GPU is too hot (e.g. an ssh one-liner polling the junction "
                         "sensor). Paces the run by the card's real state instead of a guessed "
                         "delay, so a thermal cut never fires mid-page")
    tr.add_argument("--max-chars", dest="max_chars", type=int, default=0,
                    help="split a page into sub-batches of at most this many source characters "
                         "(0 = one call per page, the production shape). Bounds how much work one "
                         "interrupted request throws away; never splits a single text")
    tr.add_argument("--resume", action="store_true",
                    help="skip pages this model already translated, so a pass cut short (a "
                         "temperature stop, a backend restart) picks up where it left off "
                         "instead of redoing the pages it already has")
    tr.add_argument("--cool-ratio", dest="cool_ratio", type=float, default=0.0,
                    help="rest this multiple of each page's own generation time before the next "
                         "one (1.0 = 50%% duty cycle). For cards without a thermal throttle, where "
                         "a slow dense model would otherwise run into the temperature cut mid-corpus")
    tr.add_argument("--timeout", type=float, default=180.0,
                    help="per-request budget in seconds; a page batch on a big model far outlasts "
                         "the SDK's production default, and tripping it would score the fallback path")
    tr.set_defaults(fn=cmd_translate)

    th = sub.add_parser("translatehtml", help="gather translate.json into one translation-scoring HTML (click-to-vote)")
    th.add_argument("--out", default=DEFAULT_OUT, help="tree with per-image translate.json (default: compare_out)")
    th.add_argument("--name", default="_compare_translate", help="output file stem under --out")
    th.add_argument("--link", action="store_true",
                    help="reference crop images by relative path instead of base64-embedding (smaller file)")
    th.set_defaults(fn=cmd_translatehtml)

    bh = sub.add_parser("boxhtml", help="gather detector <model>.png overlays into one detector-scoring HTML (click-to-vote)")
    bh.add_argument("--out", default=DEFAULT_OUT, help="tree with per-image <model>.png overlays (default: compare_out)")
    bh.add_argument("--name", default="_compare_box", help="output file stem under --out")
    bh.add_argument("--embed", action="store_true",
                    help="base64-embed overlays (default: link by relative path — full-page PNGs are large)")
    bh.set_defaults(fn=cmd_boxhtml)

    b = sub.add_parser("batch", help="run detectors over a folder tree -> <out>/<category>/<image>/<model>.png")
    b.add_argument("root", nargs="?", default="samples", help="input root of category folders (default: samples)")
    b.add_argument("--out", default=DEFAULT_OUT, help="output root (mirrors the input tree)")
    b.add_argument("--only", default=None, help="comma ids, e.g. ogkalu_rtdetr,kitsumed_seg")
    b.add_argument("--exclude", default=None, help="comma ids to drop, e.g. kitsumed_seg")
    b.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                   help="tune an adapter attr, e.g. --opt conf=0.5 --opt nms_iou=0.5 (repeatable)")
    b.add_argument("--color-by-class", dest="by_class", action="store_true",
                   help="color boxes per class (default: all red)")
    b.set_defaults(fn=cmd_batch)

    ba = sub.add_parser("ba", help="before/after dedup for one detector -> <out>/<cat>/<img>/{before,after}.png")
    ba.add_argument("root", nargs="?", default="samples", help="input root (default: samples)")
    ba.add_argument("--out", default=DEFAULT_OUT, help="output root (mirrors the input tree)")
    ba.add_argument("--detector", default=DEFAULT_REF, help="detector to A/B on dedup")
    ba.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                    help="override the AFTER dedup, e.g. --opt nms_iou=0.5 --opt contain_thresh=0.7")
    ba.add_argument("--color-by-class", dest="by_class", action="store_true",
                    help="color boxes per class (default: all red)")
    ba.set_defaults(fn=cmd_ba)

    args = ap.parse_args()
    args.fn(args)
