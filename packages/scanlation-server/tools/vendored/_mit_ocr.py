"""Runner for the vendored manga-image-translator 48px / 48px_ctc OCR (see
_mit_ocr_ctc.py, _mit_ocr_48px.py, _mit_xpos.py — models are GPL, from zyddnys).

Those models read ONE 48px-tall text line. Our rtdetr crops are whole bubbles with
several vertical columns, so this runner splits a crop into lines first (vertical
columns for portrait crops via an x ink-projection, rows for landscape), OCRs each,
and joins them in reading order (right-to-left for vertical)."""
import math
import os

import cv2
import numpy as np
import torch

_WEIGHTS = os.path.join(os.path.dirname(__file__), "_mit_weights")


def _read_dict(name: str):
    with open(os.path.join(_WEIGHTS, name), encoding="utf-8") as fp:
        return [s[:-1] for s in fp.readlines()]


def _segments(profile, min_run: int, min_gap: int, thresh_frac: float = 0.06):
    """Contiguous (start, end) runs where the ink profile is on, small gaps bridged,
    tiny runs dropped."""
    thr = max(1.0, float(profile.max()) * thresh_frac)
    on = profile > thr
    segs, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            segs.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= min_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(list(s))
    return [(a, b) for a, b in merged if b - a >= min_run]


def _split_lines(crop_rgb):
    """Return (line strips, vertical?). Portrait crop -> vertical text -> split into
    columns (right-to-left). Landscape -> horizontal -> split into rows (top-down)."""
    h, w = crop_rgb.shape[:2]
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # ink=255 (dark text)
    if binv.mean() > 127:  # ink is the majority -> text was light-on-dark; flip
        binv = 255 - binv
    ink = binv > 0
    vertical = h >= w
    if vertical:
        segs = _segments(ink.sum(axis=0), min_run=max(4, w // 30), min_gap=max(2, w // 50))
        strips = [crop_rgb[:, a:b] for a, b in segs][::-1]  # right-to-left
    else:
        segs = _segments(ink.sum(axis=1), min_run=max(4, h // 30), min_gap=max(2, h // 50))
        strips = [crop_rgb[a:b, :] for a, b in segs]  # top-to-bottom
    return (strips or [crop_rgb]), vertical


class MitOCR:
    """variant: 'ctc' (48px_ctc) or '48px' (attention)."""

    def __init__(self, variant: str):
        self.variant = variant

    def load(self, device: str) -> None:
        self.device = device
        if self.variant == "ctc":
            from ._mit_ocr_ctc import OCR
            self.dictionary = _read_dict("alphabet-all-v5.txt")
            self.model = OCR(self.dictionary, 768)
            sd = torch.load(os.path.join(_WEIGHTS, "ocr-ctc.ckpt"), map_location="cpu")
            sd = sd.get("model", sd) if isinstance(sd, dict) else sd
            for k in ("encoders.layers.0.pe.pe", "encoders.layers.1.pe.pe", "encoders.layers.2.pe.pe"):
                sd.pop(k, None)
            self.model.load_state_dict(sd, strict=False)
        else:
            from ._mit_ocr_48px import OCR
            self.dictionary = _read_dict("alphabet-all-v7.txt")
            self.model = OCR(self.dictionary, 768)
            sd = torch.load(os.path.join(_WEIGHTS, "ocr_ar_48px.ckpt"), map_location="cpu")
            sd = sd.get("model", sd) if isinstance(sd, dict) else sd
            self.model.load_state_dict(sd, strict=False)
        self.model.eval().to(device)

    def _line_region(self, strip_rgb, vertical: bool):
        a = np.rot90(strip_rgb, 1) if vertical else strip_rgb  # vertical column -> horizontal line
        a = np.ascontiguousarray(a)
        h, w = a.shape[:2]
        rw = max(1, round(w * 48 / h))
        return cv2.resize(a, (rw, 48), interpolation=cv2.INTER_AREA), rw

    def _decode(self, region, widths):
        t = ((torch.from_numpy(region).float() - 127.5) / 127.5).permute(0, 3, 1, 2).to(self.device)
        with torch.inference_mode():
            if self.variant == "ctc":
                out = []
                for line in self.model.decode(t, widths, 0):
                    chs, lps = [], []
                    for tup in line:
                        ch = self.dictionary[int(tup[0])]
                        chs.append(" " if ch == "<SP>" else ch)
                        lps.append(tup[1])
                    out.append(("".join(chs), math.exp(sum(lps) / len(lps)) if lps else 0.0))
                return out
            res = self.model.infer_beam_batch_tensor(t, widths, beams_k=5, max_seq_length=255)
            out = []
            for tup in res:
                idx, prob = tup[0], tup[1]
                chs = []
                for i in idx:
                    ch = self.dictionary[int(i)]
                    if ch in ("<S>", "</S>", "<SEP>", "<PAD>", "<EOS>", "<BOS>"):
                        continue
                    chs.append(" " if ch == "<SP>" else ch)
                out.append(("".join(chs), float(prob)))
            return out

    def recognize(self, crop_rgb) -> str:
        strips, vertical = _split_lines(crop_rgb)  # ink-projection line split
        regs = [self._line_region(s, vertical) for s in strips]
        maxw = (4 * (max(w for _, w in regs) + 7) // 4) + 128
        region = np.zeros((len(regs), 48, maxw, 3), np.uint8)
        for i, (im, w) in enumerate(regs):
            region[i, :, :w, :] = im
        results = self._decode(region, [w for _, w in regs])
        kept = [txt for txt, prob in results if txt and prob >= 0.15]
        if not kept:  # nothing cleared the bar but the model DID read something -> show best guess
            best = max((r for r in results if r[0]), key=lambda r: r[1], default=None)
            if best:
                kept = [best[0]]
        return "".join(kept)
