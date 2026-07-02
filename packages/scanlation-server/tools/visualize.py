"""Draw detected polygons + reading-order index on an image, dump deskewed crops.

    python tools/visualize.py page.png --detector ctd --out annotated.png

This is THE accuracy-debugging tool: detection is the bottleneck, so seeing the
rotated polygons land on the text (and the deskewed crops come out upright) is
how detection quality is judged by eye. Optionally pass --recognizer to also
print the OCR text per crop.
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401 - side effects: add package root to sys.path, UTF-8 stdio

from PIL import Image, ImageDraw

from app.geometry import deskew_crop
from app.pipeline import assign_reading_order
from app.registry import registry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--detector", default="ctd")
    ap.add_argument("--recognizer", default=None, help="optional: OCR each crop and print text")
    ap.add_argument("--out", default="annotated.png")
    ap.add_argument("--crops", default="crops")
    args = ap.parse_args()

    detector = registry.get("detector", args.detector)
    recognizer = registry.get("recognizer", args.recognizer) if args.recognizer else None

    img = Image.open(args.image).convert("RGB")
    regions = assign_reading_order(detector.detect(img, {}), vertical_hint=True)

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    crop_dir = Path(args.crops)
    crop_dir.mkdir(parents=True, exist_ok=True)

    for r in regions:
        poly = [(float(x), float(y)) for x, y in r.polygon]
        draw.polygon(poly, outline=(255, 0, 0), width=3)
        x0, y0, _, _ = r.bbox
        draw.text((x0 + 2, max(0, y0 - 12)), str(r.order), fill=(255, 0, 0))
        crop = deskew_crop(img, r)
        crop.save(crop_dir / f"region_{r.order:02d}.png")
        if recognizer is not None:
            text = recognizer.recognize(crop, r, {})
            print(f"[{r.order:02d}] vertical={r.vertical} angle={r.angle:.1f} -> {text!r}")

    annotated.save(args.out)
    print(f"\nwrote {args.out} + {len(regions)} crop(s) to {crop_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
