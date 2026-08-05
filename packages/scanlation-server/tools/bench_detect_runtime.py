"""detect runtime A/B — transformers(torch) vs ONNX Runtime, speed AND boxes.

forward is ~98% of detect (preprocess 4.4ms, postprocess 0.4ms), so the only
lever left on CPU is the runtime itself. The model repo ships ONNX next to the
safetensors this project installs, including an INT8 build.

Speed alone cannot decide this: the ONNX export is not bit-identical to the
safetensors, so it finds slightly different boxes. ``--html`` writes a page that
draws both runtimes' boxes on the same image (torch blue, ORT red) with the
disagreements first, because whether a missing box mattered is a judgement about
a picture, not a number.

onnxruntime is NOT a project dependency — nothing ships it, and this tool is the
only thing that wants it. Install it somewhere temporary so the container's
site-packages stays clean, and delete it after; the whole setup rebuilds in a few
minutes (the ONNX weights come from the hub on first use):

    docker cp packages/scanlation-server/tools/bench_detect_runtime.py scanlation-server:/tmp/tools/
    docker exec scanlation-server pip install --target=/tmp/ort onnxruntime
    docker exec -e PYTHONPATH=/plugins:/tmp/ort scanlation-server \
        python /tmp/tools/bench_detect_runtime.py /data/benchpages --html /data/detect_ab.html
    docker cp scanlation-server:/data/detect_ab.html .          # judge it, then
    docker exec scanlation-server rm -rf /tmp/ort /data/detect_ab.html \
        /data/hf/hub/models--ogkalu--comic-text-and-bubble-detector
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import io
import os
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import _bootstrap  # noqa: F401,E402  (adds the server package to sys.path)

from PIL import Image, ImageDraw  # noqa: E402

from scanlation_comic_text_and_bubble_detector import postprocess  # noqa: E402
from scanlation_comic_text_and_bubble_detector.plugin import ComicTextAndBubbleDetector  # noqa: E402

REPO = "ogkalu/comic-text-and-bubble-detector"
TORCH_COLOR, ORT_COLOR, MISS_COLOR = (40, 110, 255), (230, 40, 40), (255, 170, 0)


def _load_detector(device: str | None):
    det = ComicTextAndBubbleDetector()
    if device:                       # same knob /admin writes into state.json
        det._device_override = device
    det.load()
    return det


def _torch_dets(det, img, conf):
    import torch

    inputs = det._proc(images=img, return_tensors="pt").to(det._device)
    with torch.no_grad():
        out = det._model(**inputs)
    sizes = torch.tensor([img.size[::-1]]).to(det._device)   # transformers: (h, w)
    r = det._proc.post_process_object_detection(out, target_sizes=sizes, threshold=conf)[0]
    id2label = getattr(det._model.config, "id2label", {})
    return [postprocess.Det(tuple(float(v) for v in b.tolist()), id2label.get(int(l), str(int(l))), float(s))
            for s, l, b in zip(r["scores"], r["labels"], r["boxes"])]


def _ort_session(fname, threads):
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(hf_hub_download(REPO, fname), so, providers=["CPUExecutionProvider"])


def _ort_dets(sess, det, img, conf):
    # Same tensor transformers builds, so this compares runtimes and not two
    # resize implementations.
    arr = det._proc(images=img, return_tensors="pt")["pixel_values"].numpy()
    import numpy as np

    labels, boxes, scores = sess.run(None, {
        # The ONNX export wants (width, height); transformers' target_sizes is
        # (height, width). Swapping them silently rescales every box.
        "images": arr, "orig_target_sizes": np.array([[img.size[0], img.size[1]]], dtype=np.int64),
    })
    id2label = getattr(det._model.config, "id2label", {})
    return [postprocess.Det(tuple(float(v) for v in b), id2label.get(int(l), str(int(l))), float(s))
            for l, b, s in zip(labels[0], boxes[0], scores[0]) if s >= conf]


def _kept(det, dets, opts):
    dets = postprocess.filter_labels(dets, det.KEEP_LABELS)
    # The size floor is newer than some installed plugin builds — /plugins keeps
    # whatever was last installed, so don't hard-depend on it just to compare runtimes.
    small = getattr(postprocess, "filter_small", None)
    if small is not None:
        dets = small(dets, opts.get("min_area", 0), opts.get("min_side", 0))
    return postprocess.dedup(dets, opts["nms_iou"], opts["contain_thresh"])


def _pair_up(a, b):
    """Greedy IoU matching -> (pairs, only_in_a, only_in_b)."""
    rest = list(b)
    pairs, missing = [], []
    for da in a:
        best, best_iou = None, 0.0
        for db in rest:
            v = postprocess.iou(da.xyxy, db.xyxy)
            if v > best_iou:
                best, best_iou = db, v
        if best is not None and best_iou >= 0.5:
            rest.remove(best)
            pairs.append((da, best))
        else:
            missing.append(da)
    return pairs, missing, rest


def _render(img, torch_d, ort_d, pairs, only_t, only_o, width=900):
    scale = width / img.width
    canvas = img.resize((width, int(img.height * scale)), Image.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for d in torch_d:
        draw.rectangle([v * scale for v in d.xyxy], outline=TORCH_COLOR, width=2)
    for d in ort_d:
        x0, y0, x1, y1 = (v * scale for v in d.xyxy)
        draw.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], outline=ORT_COLOR, width=2)
    for d in only_t + only_o:                      # unmatched: shout about it
        draw.rectangle([v * scale for v in d.xyxy], outline=MISS_COLOR, width=6)
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()


_CSS = """
body{font:13px/1.5 system-ui,sans-serif;margin:24px;background:#111;color:#ddd}
h1{font-size:18px} table{border-collapse:collapse;width:100%}
td,th{border-top:1px solid #333;padding:10px;vertical-align:top;text-align:left}
img{max-width:100%;height:auto;border:1px solid #333}
.k{color:#888} .t{color:#4a86ff;font-weight:600} .o{color:#ff5555;font-weight:600}
.m{color:#ffaa00;font-weight:600} tr.diff{background:#1c1712} tr.same{background:#0d0d0d}
code{background:#000;padding:1px 4px;border-radius:3px}
"""


def _write_html(path, rows, summary):
    parts = [f"<style>{_CSS}</style>", "<h1>detect runtime A/B — torch vs ONNX Runtime</h1>",
             f"<p>{html.escape(summary)}</p>",
             "<p><span class='t'>파랑 = torch</span> · <span class='o'>빨강 = ONNX Runtime</span> · "
             "<span class='m'>굵은 주황 = 한쪽에만 있는 박스</span>. 불일치가 있는 페이지를 위로 정렬했다.</p>",
             "<table>"]
    for r in rows:
        cls = "diff" if (r["only_t"] or r["only_o"] or r["worst"] > 8) else "same"
        note = []
        if r["only_t"]:
            note.append(f"<span class='m'>torch에만 {len(r['only_t'])}개</span>")
        if r["only_o"]:
            note.append(f"<span class='m'>ORT에만 {len(r['only_o'])}개</span>")
        note.append(f"최대 어긋남 {r['worst']:.1f}px")
        parts.append(
            f"<tr class='{cls}'><td style='width:180px'><b>{html.escape(r['name'])}</b><br>"
            f"<span class='k'>torch {r['n_t']} / ORT {r['n_o']} 박스</span><br>"
            f"<span class='k'>torch {r['ms_t']:.0f}ms · ORT {r['ms_o']:.0f}ms</span><br>"
            + " · ".join(note) +
            f"</td><td><img src='data:image/jpeg;base64,{r['png']}'></td></tr>")
    parts.append("</table>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default=os.getenv("BENCH_DATA"), help="folder of pages, or $BENCH_DATA")
    ap.add_argument("--onnx", default="detector.onnx", help="ONNX file in the model repo")
    ap.add_argument("--threads", type=int, default=8, help="intra-op threads for BOTH runtimes")
    ap.add_argument("--conf", type=float, default=None, help="score threshold (default: the plugin's)")
    ap.add_argument("--reps", type=int, default=2, help="timed repetitions per page")
    ap.add_argument("--html", default="", help="write the box-comparison page here")
    ap.add_argument("--device", default=None,
                    help="torch device for the transformers side (e.g. cuda:1). ONNX Runtime stays on "
                         "whatever providers its build has — the stock wheel is CPU-only. On this host "
                         "cuda:0 is the gfx906 card, where torch dies in its first matmul.")
    ap.add_argument("--skip-onnx", action="store_true", help="time the torch side only")
    args = ap.parse_args()
    if not args.data:
        return print("need a page folder (or $BENCH_DATA)") or 2

    import torch
    torch.set_num_threads(args.threads)

    det = _load_detector(args.device)
    opts = det.resolve_options({})
    conf = args.conf if args.conf is not None else opts["conf"]
    sess = None if args.skip_onnx else _ort_session(args.onnx, args.threads)

    pages = sorted(p for p in glob.glob(os.path.join(args.data, "*")) if Path(p).suffix.lower() in
                   {".jpg", ".jpeg", ".png", ".webp"})
    if not pages:
        return print(f"no images under {args.data}") or 2
    print(f"{len(pages)} pages | threads={args.threads} | conf={conf} | onnx={args.onnx}")

    imgs = [(Path(p).name, Image.open(p).convert("RGB")) for p in pages]
    for _, img in imgs[:2]:                      # warm both runtimes
        _torch_dets(det, img, conf)
        if sess is not None:
            _ort_dets(sess, det, img, conf)

    rows, t_ms, o_ms, n_t_all, n_o_all, worst_all, miss_pages = [], [], [], 0, 0, 0.0, 0
    for name, img in imgs:
        tt, to = [], []
        for _ in range(args.reps):
            s = time.perf_counter(); td = _torch_dets(det, img, conf); tt.append((time.perf_counter() - s) * 1000)
            if sess is not None:
                s = time.perf_counter(); od = _ort_dets(sess, det, img, conf); to.append((time.perf_counter() - s) * 1000)
        t_ms += tt
        if sess is None:                          # torch-only run: compare against itself
            od, to = td, tt
        o_ms += to
        tk, ok = _kept(det, td, opts), _kept(det, od, opts)
        pairs, only_t, only_o = _pair_up(tk, ok)
        worst = max((max(abs(x - y) for x, y in zip(a.xyxy, b.xyxy)) for a, b in pairs), default=0.0)
        n_t_all += len(tk); n_o_all += len(ok); worst_all = max(worst_all, worst)
        if only_t or only_o:
            miss_pages += 1
        rows.append({"name": name, "n_t": len(tk), "n_o": len(ok), "worst": worst,
                     "only_t": only_t, "only_o": only_o,
                     "ms_t": statistics.median(tt), "ms_o": statistics.median(to),
                     "png": _render(img, tk, ok, pairs, only_t, only_o) if args.html else ""})
        print(f"  {name:14s} torch {statistics.median(tt):6.1f}ms/{len(tk):2d}box   "
              f"ORT {statistics.median(to):6.1f}ms/{len(ok):2d}box   worst {worst:5.1f}px"
              + ("   MISMATCH" if (only_t or only_o) else ""))

    mt, mo = statistics.median(t_ms), statistics.median(o_ms)
    summary = (f"{len(pages)} pages · torch {mt:.1f}ms vs ORT {mo:.1f}ms (x{mt / mo:.2f}) · "
               f"boxes {n_t_all} vs {n_o_all} · {miss_pages} page(s) disagree · worst corner {worst_all:.1f}px")
    print("\n" + summary)
    if args.html:
        rows.sort(key=lambda r: (not (r["only_t"] or r["only_o"]), -r["worst"]))
        _write_html(args.html, rows, summary)
        print(f"wrote {args.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
