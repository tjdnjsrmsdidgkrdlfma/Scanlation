"""Run the pipeline on one image and print the wire JSON.

    python tools/run_image.py page.png --engines ctd,dummy,dummy

--engines is detector,recognizer,translator. Use dummy for any role to isolate
the others (e.g. ctd,dummy,dummy = real detection only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python tools/...`

from PIL import Image

from app.pipeline import run_pipeline
from app.registry import registry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--src", default="ja")
    ap.add_argument("--dst", default="ko")
    ap.add_argument("--engines", default="ctd,dummy,dummy", help="detector,recognizer,translator")
    args = ap.parse_args()

    det, rec, tsl = (args.engines.split(",") + ["dummy", "dummy", "dummy"])[:3]
    detector = registry.get("detector", det)
    recognizer = registry.get("recognizer", rec)
    translator = registry.get("translator", tsl)

    img = Image.open(args.image).convert("RGB")
    result = run_pipeline(
        img, detector=detector, recognizer=recognizer, translator=translator,
        src=args.src, dst=args.dst, opt_box={}, opt_ocr={}, opt_tsl={},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n{len(result)} region(s) [{det},{rec},{tsl}]", file=sys.stderr)


if __name__ == "__main__":
    main()
