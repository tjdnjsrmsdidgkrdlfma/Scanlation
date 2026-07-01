# Dev tasks. Uses the repo-root venv (../venv). On Windows w/o make, run the
# equivalent ../venv/Scripts/python commands directly (see README).
PY ?= ../venv/Scripts/python

.PHONY: test test-all serve run-cpu viz install

test:            ## unit tests, no models/GPU
	$(PY) -m pytest -m "not slow" -q

test-all:        ## include @slow (needs CTD weights)
	$(PY) -m pytest -q

serve:           ## run the API (uvicorn, single worker)
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port 4000 --workers 1

run-cpu:         ## run pipeline on an image: make run-cpu IMG=page.png ENGINES=ctd,dummy,dummy
	$(PY) tools/run_image.py "$(IMG)" --engines "$(or $(ENGINES),ctd,dummy,dummy)"

viz:             ## visualize detection: make viz IMG=page.png
	$(PY) tools/visualize.py "$(IMG)" --detector "$(or $(DET),ctd)"

install:         ## editable install (registers entry_points)
	$(PY) -m pip install -e .
