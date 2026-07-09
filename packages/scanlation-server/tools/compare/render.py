"""Panel rendering: draw a detector's boxes on a page, tile panels into a montage."""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from compare.core import _PALETTE, DetResult


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:  # noqa: BLE001
            pass
    return ImageFont.load_default()


def _color(label: str, seen: dict[str, tuple]) -> tuple:
    if label not in seen:
        seen[label] = _PALETTE[len(seen) % len(_PALETTE)]
    return seen[label]


def render_panel(img: Image.Image, res: DetResult, title: str, *, by_class: bool = False) -> Image.Image:
    """Draw a detector's boxes on the page with a title strip above. Boxes are a
    uniform red by default (matching tools/visualize.py / the p8_after sample);
    pass by_class to color each class distinctly (e.g. rtdetr's
    bubble/text_bubble/text_free)."""
    canvas = img.convert("RGB").copy()
    d = ImageDraw.Draw(canvas)
    seen: dict[str, tuple] = {}
    counts: dict[str, int] = {}
    for b in res.boxes:
        counts[b.label] = counts.get(b.label, 0) + 1
        col = _color(b.label, seen) if by_class else (255, 0, 0)
        if b.polygon and len(b.polygon) >= 3:
            d.polygon([(float(x), float(y)) for x, y in b.polygon], outline=col, width=3)
        else:
            d.rectangle(b.xyxy, outline=col, width=3)
    strip_h = 34
    out = Image.new("RGB", (canvas.width, canvas.height + strip_h), (20, 20, 20))
    out.paste(canvas, (0, strip_h))
    dd = ImageDraw.Draw(out)
    legend = "  ".join(f"{k}={n}" for k, n in counts.items()) or "0"
    ms = f"  {res.ms:.0f}ms" if res.ms else ""  # omit a meaningless 0ms (e.g. the ba mode)
    dd.text((6, 7), f"{title}   {len(res.boxes)} boxes{ms}   [{legend}]",
            fill=(255, 255, 255), font=_font(20))
    return out


def montage(panels: list[Image.Image], out_path: Path, width: int = 720) -> None:
    if not panels:
        return
    scaled = [p.resize((width, int(p.height * width / p.width))) for p in panels]
    ncols = min(len(scaled), 2 if len(scaled) <= 4 else 3)
    nrows = math.ceil(len(scaled) / ncols)
    cw, ch, gap = width, max(p.height for p in scaled), 10
    grid = Image.new("RGB", (ncols * cw + (ncols + 1) * gap, nrows * ch + (nrows + 1) * gap), (60, 60, 60))
    for i, p in enumerate(scaled):
        r, c = divmod(i, ncols)
        grid.paste(p, (gap + c * (cw + gap), gap + r * (ch + gap)))
    grid.save(out_path)
