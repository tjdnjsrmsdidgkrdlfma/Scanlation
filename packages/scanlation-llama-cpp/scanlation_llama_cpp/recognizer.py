"""LlamaCppRecognizer — text recognition via an OpenAI-compatible VISION endpoint.

Serves the same VLM recognizer as the in-process engine, but from llama.cpp's
`llama-server` instead of transformers in this process. Measured on this project's
own crops (``tools/recognize-decode-bound.md``): **~10x the per-crop throughput at
~1/4 the VRAM**, with the text effectively unchanged (35/42 identical-or-cosmetic,
3 fixes, 4 regressions).

The win is structural, not tuning: per-crop time was ~95% autoregressive decode at
a flat ~64ms/token of which weight-read is ~4% and compute <1% — i.e. host-side
overhead of the eager transformers loop, which llama.cpp has no Python/torch
dispatch layer to pay. It decodes at 2.4ms/token here — ~65% of the card's memory
bandwidth, where transformers reached 4.6% — so what is left is the physical floor
of a B=1 decode. It also holds ONE model copy server-side for all callers, where
the worker pool holds one per worker.

The 10x needs the server pinned to its GPU by ``GGML_VK_VISIBLE_DEVICES``, not by
``--device``: the latter constrains only the LM layers and lets the vision encoder
pick another card. See the doc's §6 and the unit examples under ``deploy/``.

Leave this engine's worker count at 1 unless it has been re-measured — concurrency
was only ever timed on a mis-pinned server, so the old "c>1 buys nothing" number
does not describe this configuration.

Model-agnostic by design: it sends an image + prompt to /v1/chat/completions, so
any OpenAI-compatible vision server works (llama.cpp, vLLM, SGLang…).

What ``/admin`` exposes is deliberately narrow. A recognizer's contract is crop in ->
text out, so ``OPTION_SCHEMA`` carries only what changes how the CROP is read
(``max_pixels``, ``downscale_mode``). The serving details — which model, the prompt it
was trained on, the token ceiling, the decode temperature — belong to whoever runs the
llama-server, so they are env-settable (the functions below) and stay off the operator's
form, the same way the per-crop timeout does.
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


def recognize_model() -> str:
    """Model id sent with each crop — LLAMACPP_RECOGNIZE_MODEL (default unset, and then
    omitted from the request entirely). llama-server serves the one model it was launched
    with (-hf) and ignores the field; other OpenAI servers require it. It names a
    server-side deployment, so it rides with the endpoint env, not the admin form."""
    return os.getenv("LLAMACPP_RECOGNIZE_MODEL", "")


def recognize_prompt() -> str:
    """Instruction sent with the crop — SCANLATION_RECOGNIZE_PROMPT (default "OCR:", what
    PaddleOCR-VL is trained on). It is a property of WHICH model the server holds, so it
    moves with that deployment; a differently-prompted model needs it changed there."""
    return os.getenv("SCANLATION_RECOGNIZE_PROMPT", "OCR:")


def recognize_max_tokens() -> int:
    """Output-token ceiling per crop — SCANLATION_RECOGNIZE_MAX_TOKENS (default 1024).
    A runaway brake on generation, not a quality knob."""
    return int(os.getenv("SCANLATION_RECOGNIZE_MAX_TOKENS", "1024"))


def recognize_temperature() -> float:
    """Decode temperature — SCANLATION_RECOGNIZE_TEMPERATURE (default 0 = greedy).
    Recognition wants one deterministic reading of the pixels."""
    return float(os.getenv("SCANLATION_RECOGNIZE_TEMPERATURE", "0"))


class LlamaCppRecognizer(HttpEngineBase):
    name = "llama.cpp"
    # The id stays "llama.cpp" (registry key + persisted selection), but the LABEL
    # says which kind of server this is: the plugin's two engines otherwise render
    # as the same word in /admin's role dropdowns and the popup, and every sibling
    # recognizer is named after its MODEL, which llama.cpp can't be (the GGUF is
    # picked by the llama-server command line).
    display_name = "llama.cpp (vision)"
    homepage = "https://github.com/ggml-org/llama.cpp"
    description = ("Text recognition via an OpenAI-compatible /v1 vision server (llama.cpp, vllm…; "
                   "must be running with a VLM + its mmproj). ~10x faster and ~1/4 the VRAM of the "
                   "in-process engine.")
    # Its own endpoint: the recognizer serves a DIFFERENT model than the translator,
    # so it is a second llama-server instance on its own port, not the same one.
    ENDPOINT_ENV = "LLAMACPP_RECOGNIZE_ENDPOINT"
    DEFAULT_ENDPOINT = "http://127.0.0.1:8090"
    ROLE_LABEL = "recognizer"
    # Only the crop-reading knobs — the serving details are env (module top).
    OPTION_SCHEMA = {
        # 300k/box, not the in-process engine's 150k/pow2: pow2 only halves, so a crop
        # just over the cap drops to a QUARTER of it — a 302k crop read at 75k lost 3 of
        # its 4 lines. box lands on the cap exactly, and the cap stays as the guard rail
        # that keeps a huge crop from blowing up the tail (doc §7).
        "max_pixels": {"type": int,
                       "default": int(os.environ.get("SCANLATION_RECOGNIZE_MAX_PIXELS", "300000")),
                       "description": "Downscale crops above this many pixels before OCR to cut vision tokens. 0 = off."},
        "downscale_mode": {"type": str,
                           "default": os.environ.get("SCANLATION_RECOGNIZE_DOWNSCALE_MODE", "box"),
                           "choices": list(DOWNSCALE_MODES),
                           "description": "How to downscale when max_pixels applies (box recommended — pow2 overshoots)."},
    }
    SUPPORTED_SRC = ["ja", "en", "zh", "ko"]

    def _timeout(self) -> float:
        return recognize_http_timeout()

    def _models_url(self) -> str:
        return f"{self.endpoint}/v1/models"

    def _parse_models(self, payload: dict) -> list[str]:
        return [m["id"] for m in payload.get("data", []) if m.get("id")]

    def _body(self, crop: Image.Image) -> dict:
        """The vision chat-completions body: one user message carrying the crop as a
        data: URI plus the instruction. PNG (lossless) because a JPEG's ringing lands
        exactly on the thin strokes and small kana this model is read on."""
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": recognize_prompt()},
            ]}],
            "temperature": recognize_temperature(),
            "max_tokens": recognize_max_tokens(),
            "stream": False,
        }
        model = recognize_model()
        if model:  # omit entirely rather than send "" — llama-server ignores it anyway
            body["model"] = model
        return body

    def _extract(self, data: dict) -> str:
        return ((data.get("choices") or [{}])[0].get("message", {}).get("content", "") or "").strip()

    def recognize(self, crop: Image.Image, region: Region, options: dict[str, Any]) -> str:
        """One crop -> its text. ``crop`` is already deskewed upright by the pipeline;
        ``region`` is unused (recognition reads the pixels, not the geometry). The cap
        is applied HERE, before upload, so it also shrinks what goes over the wire."""
        options = self.resolve_options(options)
        crop = downscale_to_cap(to_rgb(crop), options["max_pixels"], options["downscale_mode"])
        return self._extract(self._post("/v1/chat/completions", self._body(crop)))
