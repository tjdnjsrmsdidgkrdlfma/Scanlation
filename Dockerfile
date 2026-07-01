# Scanlation server — core-only image.
#
# Ships scanlation-sdk + scanlation-server (the `dummy` engine only). The real
# engine plugins (ctd/mangaocr/ollama/llamacpp) are NOT installed here: their
# *source* sits under /opt/engines and the admin page pip-installs the chosen
# ones into the /plugins volume at runtime ("설치한 패키지 = 탑재 엔진" — installing the
# package is how an engine appears, inside the container too). LLM backends
# (ollama / llama.cpp) run outside; the translator plugins just point at them.
#
# Build context is the repo root:  docker build -t scanlation-server .
FROM python:3.11-slim

# Generic OS runtime libs that engine backends (onnxruntime / torch / opencv)
# dlopen when loaded. These are shared C libraries, not plugin packages — the
# core (opencv-headless deskew) needs libgomp/libglib too.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SCANLATION_BASE_DIR=/data \
    SCANLATION_MODELS_DIR=/data/models \
    SCANLATION_ENGINES_SRC=/opt/engines \
    SCANLATION_SDK_SRC=/opt/engines/scanlation-sdk \
    SCANLATION_PLUGINS_DIR=/plugins \
    PYTHONPATH=/plugins \
    HF_HOME=/data/hf \
    SCANLATION_DEVICE=cpu

# Engine + sdk SOURCE (unbuilt, zero deps installed). scanlation-sdk lives here
# too so it can be co-installed with each engine at runtime (pip can't fetch the
# local scanlation-sdk from any index); the catalog scanner skips sdk/server.
COPY packages/scanlation-sdk      /opt/engines/scanlation-sdk
COPY packages/scanlation-ctd      /opt/engines/scanlation-ctd
COPY packages/scanlation-mangaocr /opt/engines/scanlation-mangaocr
COPY packages/scanlation-ollama   /opt/engines/scanlation-ollama
COPY packages/scanlation-llamacpp /opt/engines/scanlation-llamacpp
COPY packages/scanlation-server   /opt/scanlation-server

# Core only: sdk + server. Editable so the packaged web/ (admin SPA) + tools/
# assets resolve from the source tree. The dummy engine ships in the server pkg.
RUN pip install -e /opt/engines/scanlation-sdk -e /opt/scanlation-server

# Non-root; /data (state, sqlite, weights, HF cache) and /plugins (runtime-
# installed engine packages) are volumes it owns and writes to.
RUN useradd -m -u 10001 app \
 && mkdir -p /data /plugins \
 && chown -R app:app /data /plugins /opt
USER app

EXPOSE 4000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "4000"]
