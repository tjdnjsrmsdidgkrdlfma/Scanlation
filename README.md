# Scanlation

Self-owned, contract-based manga **OCR + in-place translation** stack (Japanese → Korean),
replacing the Crivella `ocr_translate` + browser-extension setup. Accuracy-first; the
real bottleneck in manga is **detection/segmentation**, so the contract carries *rotated*
geometry (4-point polygons + angle + mask) and the pipeline deskews tilted text (SFX,
vertical JP) before recognition.

Design rationale & full roadmap: [`YOMU_DESIGN.md`](YOMU_DESIGN.md).

## Status

Server stages **P0–P3** are implemented and tested:

| Stage | What | Test |
|---|---|---|
| P0 | FastAPI skeleton, handshake, CORS | `GET /` |
| P1 | contracts · registry · state · cache · pipeline · **all wire routes** · dummy engines | `test_routes`, `test_pipeline`, `test_contracts` |
| P2 | deskew geometry (OpenCV homography, axis-aligned fast path) | `test_geometry` |
| P3 | `CTDDetector` (comic-text-detector ONNX) · mask→quad decode · `visualize.py` | `test_ctd_decode`, `test_ctd` (slow) |

Wire protocol is a drop-in for the existing `ocr_extension` client (verified against source):
md5 is over the **base64 string**; box is `[x_min, y_min, x_max, y_max]` (client reads `[l,b,r,t]`);
`/run_ocrtsl/` does the lazy(md5-only)→work(contents) flow.

Not yet built: P4 manga-ocr recognizer, P5 ollama translator, P6 MV3 extension, P7 Docker/ROCm.

## Layout

```
server/
  app/        contracts, geometry, pipeline, registry, cache, state, config, routes/
  plugins/    dummy/ (test doubles), detector_ctd/ (CTD ONNX)
  tools/      run_image.py, visualize.py
  tests/
```

## Dev setup (CPU, no GPU needed)

Deps are isolated in a repo-root venv (`.venv`, git-ignored):

```bash
python -m venv .venv
./.venv/Scripts/python -m pip install -e ./server[ctd,dev]   # or install deps directly
```

Common commands (from `server/`; `make` targets wrap these):

```bash
../.venv/Scripts/python -m pytest -m "not slow"        # unit tests, no models
../.venv/Scripts/python -m uvicorn app.main:app --port 4000 --workers 1
../.venv/Scripts/python tools/run_image.py page.png --engines ctd,dummy,dummy
../.venv/Scripts/python tools/visualize.py page.png --detector ctd --out annotated.png
```

`--engines detector,recognizer,translator` — set any role to `dummy` to isolate the others.

## CTD weights

Not bundled. Set `SCANLATION_CTD_MODEL=/path/model.onnx` or drop an `.onnx` into
`server/models/ctd/` (e.g. `mayocream/comic-text-detector-onnx`). Until present, `visualize.py`
and the slow CTD test are skipped. Decoding (mask→rotated quads) must be verified by eye via
`visualize.py` on a real page — that's the P3 acceptance check.
