"""LlamaCppTranslator — translation via an OpenAI-compatible chat endpoint.

Primary use: run the LLM under llama.cpp's `llama-server` with the **Vulkan**
backend, which is often more reliable than ROCm on newer AMD GPUs (e.g. gfx1200
/ RDNA4) where ollama's Vulkan support is weak. Same system prompt + template as
the ollama backend (shared via scanlation_sdk.prompt), so translations are
consistent regardless of which backend the GPU happens to like.

Talks `/v1/chat/completions`, so it also works with any OpenAI-compatible server
(vllm, LM Studio, ollama's own /v1, etc.). Reasoning-model `<think>...</think>`
spans are stripped (llama.cpp has no ollama-style think:false toggle). The client
lifecycle + guardrails live in HttpTranslatorBase; only the request body shape +
response parsing are here.
"""
from __future__ import annotations

import re

from scanlation_sdk.http_translator import HttpTranslatorBase

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LlamaCppTranslator(HttpTranslatorBase):
    name = "llamacpp"
    display_name = "llama.cpp"
    homepage = "https://github.com/ggml-org/llama.cpp"
    description = "Translation via an OpenAI-compatible /v1/chat/completions server (llama.cpp Vulkan, vllm, LM Studio...)."
    warning = "Requires a running server (LLAMACPP_ENDPOINT, default http://127.0.0.1:8080) with a model loaded."
    ENDPOINT_ENV = "LLAMACPP_ENDPOINT"
    DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
    OPTION_SCHEMA = {
        "model": {"type": str, "default": "", "description": "Model id (from the server's /v1/models). Required — pick it in /admin. (llama-server ignores it; other OpenAI servers require it.)"},
        "temperature": {"type": float, "default": 0.0, "description": "Sampling temperature (0 = deterministic)."},
        "seed": {"type": int, "default": 42, "description": "RNG seed."},
        "top_p": {"type": float, "default": 1.0, "description": "Nucleus sampling p."},
        "max_tokens": {"type": int, "default": 512, "description": "Max tokens to generate."},
        "strip_think": {"type": bool, "default": True, "description": "Remove <think>...</think> from reasoning models."},
    }

    def _chat(self, body: dict) -> dict:
        """POST /v1/chat/completions. Kept as the unit-test seam (tests fake this)."""
        return self._post("/v1/chat/completions", body)

    def _models_url(self) -> str:
        return f"{self.endpoint}/v1/models"

    def _parse_models(self, payload: dict) -> list[str]:
        return [m["id"] for m in payload.get("data", []) if m.get("id")]

    def _body(self, model: str, system: str, prompt: str, options: dict, max_tokens_default: int) -> dict:
        """The shared chat-completions body. max_tokens default differs by path:
        512 for one bubble, more for a whole image's batch output."""
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(options.get("temperature", 0.0)),
            "top_p": float(options.get("top_p", 1.0)),
            "seed": int(options.get("seed", 42)),
            "max_tokens": int(options.get("max_tokens", max_tokens_default)),
            "stream": False,
        }

    def _extract(self, data: dict, options: dict) -> str:
        out = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        if options.get("strip_think", True):
            out = _THINK.sub("", out)
        return out.strip()

    def _translate(self, model: str, system: str, prompt: str, options: dict) -> str:
        return self._extract(self._chat(self._body(model, system, prompt, options, 512)), options)

    def _translate_batch_call(self, model: str, system: str, prompt: str, schema: dict, options: dict) -> str:
        """Batch: same body plus response_format=json_schema to force the exact
        JSON shape, and a larger max_tokens default (whole-image output)."""
        body = self._body(model, system, prompt, options, 1024)
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "translations", "schema": schema, "strict": True},
        }
        return self._extract(self._chat(body), options)
