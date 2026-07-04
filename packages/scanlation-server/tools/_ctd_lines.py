"""CTD line splitter for the 48px / 48px_ctc OCR models.

The 48px models read ONE horizontal 48px-tall text line — they were built to
consume the line quads a text DETECTOR emits, not a whole bubble (manga-image-
translator pairs comic-text-detector + 48px exactly this way). Our crude ink-
projection splitter (_mit_ocr._split_lines) stands in for that detector and is
fragile on tilted / touching / single-column crops.

This runs comic-text-detector.onnx on a crop and decodes its text-segmentation
mask into rotated per-COLUMN quads (a morph-close kernel taller than wide bridges
glyphs DOWN a vertical JP column while keeping neighbouring columns apart), warps
each column flat to a horizontal strip, and returns them in reading order
(right-to-left for vertical). Drop-in replacement for _split_lines.

Weights: <scanlation-server>/models/ctd/comic-text-detector.onnx (or set
SCANLATION_CTD_MODEL). Geometry condensed from the removed scanlation-ctd plugin
(decode.mask_to_regions), git 4ff8621~1, with the scanlation_sdk deps stripped.
"""
import glob
import os

import cv2
import numpy as np

# mask-decode tuning (same defaults as the old plugin's decode.DEFAULTS)
_DEF = dict(mask_threshold=0.3, min_area=200, min_side=12, unclip_ratio=1.2,
            merge_px=16, merge_aspect=1.7)


def _model_path() -> str:
    env = os.environ.get("SCANLATION_CTD_MODEL")
    if env and os.path.isfile(env):
        return env
    d = os.path.join(os.path.dirname(__file__), "..", "models", "ctd")
    hits = sorted(glob.glob(os.path.join(d, "*.onnx")))
    if not hits:
        raise RuntimeError(f"comic-text-detector.onnx not in {os.path.abspath(d)} "
                           "(or set SCANLATION_CTD_MODEL)")
    return hits[0]


def _letterbox(img, size: int, pad_value: int = 114):
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    nh, nw = int(round(h * ratio)), int(round(w * ratio))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pw, ph = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, img.shape[2]), pad_value, dtype=img.dtype)
    out[ph:ph + nh, pw:pw + nw] = resized
    return out, ratio, (pw, ph)


def _order_pts(pts):
    """Corners as (tl, tr, br, bl) via the sum/diff heuristic."""
    pts = pts.astype(np.float32)
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]], np.float32)


def _warp(img, quad):
    """Perspective-warp a (possibly rotated) quad to an axis-aligned strip."""
    tl, tr, br, bl = _order_pts(quad)
    W = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    H = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if W < 2 or H < 2:
        return None
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(_order_pts(quad), dst)
    return cv2.warpPerspective(img, M, (W, H))


class CtdLines:
    """Run per crop -> list of horizontal line strips (RGB) in reading order."""

    def __init__(self, **opt):
        self.opt = {**_DEF, **opt}
        self._sess = None
        self._in = None
        self._size = 1024

    def load(self, device: str = "cpu"):
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
        gpu = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider") if p in avail]
        prov = (gpu if device != "cpu" else []) + ["CPUExecutionProvider"]
        self._sess = ort.InferenceSession(_model_path(), providers=prov)
        self._in = self._sess.get_inputs()[0].name
        shp = self._sess.get_inputs()[0].shape
        if isinstance(shp[-1], int) and shp[-1] > 0:  # static input size -> must feed exactly that
            self._size = int(shp[-1])
        return self

    @staticmethod
    def _pick_mask(outs):
        """Text seg mask = the 4-D output with the largest area, fewest channels."""
        best, bkey = None, None
        for o in outs:
            if o.ndim != 4:
                continue
            _, c, h, w = o.shape
            key = (h * w, -c)
            if bkey is None or key > bkey:
                best, bkey = o, key
        m = np.asarray(best)[0, 0]
        if m.min() < 0 or m.max() > 1:  # logits -> sigmoid
            m = 1.0 / (1.0 + np.exp(-m))
        return m.astype(np.float32)

    def _quads(self, mask, ratio, pad, ow, oh):
        o = self.opt
        binary = (mask >= o["mask_threshold"]).astype(np.uint8) * 255
        if o["merge_px"] > 0:  # close gaps DOWN a column (kh>kw) without fusing columns
            kw = max(1, int(round(o["merge_px"])))
            kh = max(1, int(round(o["merge_px"] * max(o["merge_aspect"], 1.0))))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        inv = 1.0 / ratio if ratio else 1.0
        pw, ph = pad
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
            owid, ohei = rw * inv, rh * inv
            if min(owid, ohei) < o["min_side"] or owid * ohei < o["min_area"]:
                continue
            quad = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.float32)
            quad[:, 0] = np.clip((quad[:, 0] - pw) * inv, 0, ow - 1)
            quad[:, 1] = np.clip((quad[:, 1] - ph) * inv, 0, oh - 1)
            out.append((quad, rh > rw))  # (quad in crop px, vertical?)
        return out

    def lines(self, crop_rgb):
        if self._sess is None:
            self.load()
        img = np.ascontiguousarray(crop_rgb)
        oh, ow = img.shape[:2]
        padded, ratio, pad = _letterbox(img, self._size)
        blob = np.transpose(padded.astype(np.float32) / 255.0, (2, 0, 1))[None]
        outs = self._sess.run(None, {self._in: blob})
        mask = self._pick_mask(outs)
        if mask.shape[:2] != (self._size, self._size):
            mask = cv2.resize(mask, (self._size, self._size), interpolation=cv2.INTER_LINEAR)
        quads = self._quads(mask, ratio, pad, ow, oh)
        if not quads:
            return []
        vertical = sum(1 for _, v in quads if v) >= len(quads) / 2

        def ctr(q):
            return float(q[:, 0].mean()), float(q[:, 1].mean())
        # reading order: vertical JP -> right-to-left; else top-down then left-right
        quads.sort(key=lambda qv: (-ctr(qv[0])[0],) if vertical else (ctr(qv[0])[1], ctr(qv[0])[0]))
        strips = []
        for quad, _v in quads:
            s = _warp(img, quad)
            if s is None:
                continue
            if s.shape[0] > s.shape[1]:  # portrait strip = vertical line -> lay flat (CCW; validated for 48px)
                s = np.rot90(s, 1)
            strips.append(np.ascontiguousarray(s))
        return strips
