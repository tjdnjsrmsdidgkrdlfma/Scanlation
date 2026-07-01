# Dev tasks. Uses the repo-root venv (../../venv). On Windows w/o make, run the
# equivalent ../../venv/Scripts/python commands directly (see README).
PY ?= ../../venv/Scripts/python

.PHONY: test test-ctd test-mangaocr serve viz install

test:            ## fast unit suite, no models/GPU (hand-rolled runner)
	$(PY) -m tests

test-ctd:        ## CTD model smoke (needs onnx weights; self-skips otherwise)
	$(PY) -m tests.test_ctd

test-mangaocr:   ## manga-ocr smoke (needs the package; self-skips otherwise)
	$(PY) -m tests.test_mangaocr

serve:           ## run the API (uvicorn, single worker)
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port 4000 --workers 1

viz:             ## visualize detection: make viz IMG=page.png
	$(PY) tools/visualize.py "$(IMG)" --detector "$(or $(DET),ctd)"

install:         ## editable core (sdk + server; no engine — engines are separate pkgs)
	$(PY) -m pip install -e ../scanlation-sdk -e .
