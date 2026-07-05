# Scanlation server — core-only image.
#
# The image contains ONLY the core: scanlation-sdk + scanlation-server (the
# `dummy` engine). No engine plugin code is baked in. The real engines
# (comic-text-and-bubble-detector/manga-ocr/Ollama/llama.cpp) are pip-installed from GitHub at runtime when
# you click "install" in /admin — into the /plugins volume, so they survive
# container recreation and their heavy deps (onnxruntime/torch) arrive only then.
# LLM backends (ollama / llama.cpp) run outside; the translator plugins point at
# them via env.
#
# Build context is the repo root:  docker build -t scanlation-server .
FROM python:3.11-slim

# git   — runtime plugin install fetches engines via `pip install git+...`.
# libgomp1/libglib2.0-0 — generic C runtime libs the engine backends
#   (onnxruntime/torch/opencv) dlopen; the core (opencv-headless deskew) needs them too.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgomp1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SCANLATION_BASE_DIR=/data \
    SCANLATION_MODELS_DIR=/data/models \
    SCANLATION_PLUGINS_DIR=/plugins \
    PYTHONPATH=/plugins \
    HF_HOME=/data/hf \
    SCANLATION_ENGINE_REPO=https://github.com/tjdnjsrmsdidgkrdlfma/Scanlation.git \
    SCANLATION_ENGINE_REF=main

# Core only: sdk + server, installed non-editable (a real package, not a mounted
# source tree). Build from the copied source, then discard it — the final image
# carries no repo working tree, just the installed core (+ dummy engine).
COPY packages/scanlation-sdk    /tmp/build/scanlation-sdk
COPY packages/scanlation-server /tmp/build/scanlation-server
RUN pip install /tmp/build/scanlation-sdk /tmp/build/scanlation-server \
 && rm -rf /tmp/build

# Non-root; /data (state, sqlite, weights, HF cache) and /plugins (runtime-
# installed engine packages) are volumes it owns and writes to.
RUN useradd -m -u 10001 app \
 && mkdir -p /data /plugins \
 && chown -R app:app /data /plugins
USER app

EXPOSE 4000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "4000"]
