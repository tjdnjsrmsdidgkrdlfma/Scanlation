"""LlamaCppRecognizer — text recognition via an OpenAI-compatible VISION endpoint.

Serves the same VLM recognizer as the in-process engine, but from llama.cpp's
`llama-server` instead of transformers in this process. Measured on this project's
own crops (``tools/recognize-decode-bound.md``): **~2.2x the per-crop throughput at
~1/4 the VRAM**, with the text effectively unchanged (35/42 identical-or-cosmetic,
3 fixes, 4 regressions).

The win is structural, not tuning: per-crop time was ~95% autoregressive decode at
a flat ~64ms/token of which weight-read is ~9% and compute <1% — i.e. host-side
overhead of the eager transformers loop, which llama.cpp has no Python/torch
dispatch layer to pay. It also holds ONE model copy server-side for all callers,
where the worker pool holds one per worker.

Do NOT reach for concurrency here: with the host overhead gone the GPU is already
saturated, so extra in-flight requests bought 1.06x while tripling per-crop latency
(same doc). Leave this engine's worker count at 1.

Model-agnostic by design: it sends an image + prompt to /v1/chat/completions, so
any OpenAI-compatible vision server works (llama.cpp, vLLM, SGLang…). WHICH model
is served is a deployment choice (the llama-server command line), not a plugin
setting — mirroring how the translator half already works.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Any

from PIL import Image

from scanlation_sdk.contracts import Region
from scanlation_sdk.http_engine import HttpEngineBase
from scanlation_sdk.local_engine import DOWNSCALE_MODES, downscale_to_cap, to_rgb


def recognize_http_timeout() -> float:
    """Per-crop budget (seconds) — SCANLATION_RECOGNIZE_HTTP_TIMEOUT (default 120).
    Far longer than the translator's 10s: a recognize runs ~1s/crop and the FIRST
    call to a cold server also absorbs its model load, so a short timeout would turn
    a healthy startup into a failed page."""
    return float(os.getenv("SCANLATION_RECOGNIZE_HTTP_TIMEOUT", "120"))


class LlamaCppRecognizer(HttpEngineBase):
    name = "llama.cpp"
    display_name = "llama.cpp"
    homepage = "https://github.com/ggml-org/llama.cpp"
    description = ("Text recognition via an OpenAI-compatible /v1 vision server (llama.cpp, vllm…; "
                   "must be running with a VLM + its mmproj). ~2.2x faster and ~1/4 the VRAM of the "
                   "in-process engine.")
    # Its own endpoint: the recognizer serves a DIFFERENT model than the translator,
    # so it is a second llama-server instance on its own port, not the same one.
    ENDPOINT_ENV = "LLAMACPP_RECOGNIZE_ENDPOINT"
    DEFAULT_ENDPOINT = "http://127.0.0.1:8090"
    ROLE_LABEL = "recognizer"
    OPTION_SCHEMA = {
        "model": {"type": str, "default": "",
                  "description": "Model id (from the server's /v1/models). Optional — llama-server serves "
                                 "whatever it was started with and ignores this; other OpenAI servers require it."},
        "prompt": {"type": str, "default": "OCR:",
                   "description": "Instruction sent with the crop. 'OCR:' is what PaddleOCR-VL is trained on — "
                                  "change only for a differently-prompted model."},
        "max_tokens": {"type": int, "default": 1024,
                       "description": "Max output tokens per crop; lower to cap runaway generation."},
        "temperature": {"type": float, "default": 0.0,
                        "description": "0 = greedy/deterministic, which is what OCR wants."},
        # The cap matters MORE here than in-process: llama.cpp's vision cost is linear in
        # pixels above a floor (~4.7us/px measured), so uncapped crops cost 1.4x. But it
        # must land AT the cap, not under it -- see downscale_mode.
        "max_pixels": {"type": int,
                       "default": int(os.environ.get("SCANLATION_RECOGNIZE_MAX_PIXELS", "150000")),
                       "description": "Downscale crops above this many pixels before OCR to cut vision tokens. "
                                      "150k sits just above the model's own floor, where resolution is still free. 0 = off."},
        # `box` here, not the in-process engine's `pow2`. pow2 only halves, so it
        # overshoots the cap (a 259k crop lands at 65k) -- and llama.cpp then scales that
        # back UP to its clip.vision.image_min_pixels floor (147384). Same token count,
        # same speed, 2.3x less real detail, and a crop that truncated because of it.
        # `box` lands exactly at the cap with the same non-ringing area-average filter.
        "downscale_mode": {"type": str,
                           "default": os.environ.get("SCANLATION_RECOGNIZE_DOWNSCALE_MODE", "box"),
                           "description": f"How to downscale when max_pixels applies: {' / '.join(DOWNSCALE_MODES)}. "
                                          "box (recommended here) lands exactly at the cap; pow2 undershoots it and the "
                                          "server just upscales again, losing detail for nothing."},
    }
    SUPPORTED_SRC = ["ja", "en", "zh", "ko"]

    def _timeout(self) -> float:
        return recognize_http_timeout()

    def _models_url(self) -> str:
        return f"{self.endpoint}/v1/models"

    def _parse_models(self, payload: dict) -> list[str]:
        return [m["id"] for m in payload.get("data", []) if m.get("id")]

    def _body(self, crop: Image.Image, options: dict) -> dict:
        """The vision chat-completions body: one user message carrying the crop as a
        data: URI plus the instruction. PNG (lossless) because a JPEG's ringing lands
        exactly on the thin strokes and small kana this model is read on."""
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": options["prompt"]},
            ]}],
            "temperature": options["temperature"],
            "max_tokens": options["max_tokens"],
            "stream": False,
        }
        if options["model"]:  # omit entirely rather than send "" — llama-server ignores it anyway
            body["model"] = options["model"]
        return body

    def _extract(self, data: dict) -> str:
        return ((data.get("choices") or [{}])[0].get("message", {}).get("content", "") or "").strip()

    def recognize(self, crop: Image.Image, region: Region, options: dict[str, Any]) -> str:
        """One crop -> its text. ``crop`` is already deskewed upright by the pipeline;
        ``region`` is unused (recognition reads the pixels, not the geometry). The cap
        is applied HERE, before upload, so it also shrinks what goes over the wire."""
        options = self.resolve_options(options)
        crop = downscale_to_cap(to_rgb(crop), options["max_pixels"], options["downscale_mode"])
        return self._extract(self._post("/v1/chat/completions", self._body(crop, options)))
