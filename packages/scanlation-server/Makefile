# Dev tasks. Uses the repo-root venv (../../venv). On Windows w/o make, run the
# equivalent ../../venv/Scripts/python commands directly (see README).
PY ?= ../../venv/Scripts/python

.PHONY: test serve viz install

test:            ## fast unit suite, no models/GPU (hand-rolled runner)
	$(PY) -m tests
	# engine smokes (comic-text-and-bubble-detector/manga-ocr/...) live in their own packages: run
	# `../../venv/Scripts/python -m tests` from packages/scanlation-<engine>/

serve:           ## run the API (uvicorn, single worker)
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port 4010 --workers 1

viz:             ## visualize detection: make viz IMG=page.png
	$(PY) tools/visualize.py "$(IMG)" --detector "$(or $(DET),comic-text-and-bubble-detector)"

install:         ## editable core (sdk + server; no engine — engines are separate pkgs)
	$(PY) -m pip install -e ../scanlation-sdk -e .
