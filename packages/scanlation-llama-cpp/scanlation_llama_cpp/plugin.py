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

from scanlation_sdk.http_translator import COMMON_LLM_OPTIONS, HttpTranslatorBase

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LlamaCppTranslator(HttpTranslatorBase):
    name = "llama.cpp"
    display_name = "llama.cpp"
    homepage = "https://github.com/ggml-org/llama.cpp"
    description = "LLM translation via an OpenAI-compatible /v1 server (llama.cpp, vllm, LM Studio…; must be running, model selected in /admin)."
    ENDPOINT_ENV = "LLAMACPP_ENDPOINT"
    DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
    OPTION_SCHEMA = {
        "model": {"type": str, "default": "", "description": "Model id (from the server's /v1/models). Required — pick it in /admin. (llama-server ignores it; other OpenAI servers require it.)"},
        **COMMON_LLM_OPTIONS,  # temperature, seed, top_p
        "think": {"type": bool, "default": False, "description": "Enable model 'thinking'/reasoning (slower; off for speed). Sent as chat_template_kwargs.enable_thinking — the model's chat template must honor it; else disable globally via the server's --reasoning-budget 0."},
        "strip_think": {"type": bool, "default": True, "description": "Remove <think>...</think> from reasoning models."},
    }

    def _models_url(self) -> str:
        return f"{self.endpoint}/v1/models"

    def _parse_models(self, payload: dict) -> list[str]:
        return [m["id"] for m in payload.get("data", []) if m.get("id")]

    def _body(self, model: str, system: str, prompt: str, options: dict) -> dict:
        """The shared chat-completions body. Options arrive already resolved against
        OPTION_SCHEMA (defaults filled + typed) by resolve_options, so this just
        reads them. No explicit max_tokens — the single path stops at EOS and the
        batch's JSON grammar bounds its output, matching the ollama backend (which
        sends no output cap either)."""
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": options["temperature"],
            "top_p": options["top_p"],
            "seed": options["seed"],
            "stream": False,
            # Thinking/reasoning toggle -> the model's chat template (Qwen3-style
            # `enable_thinking`). Off by default: a reasoning model otherwise emits a
            # long hidden <think> before the answer (much slower). Template-dependent;
            # if ignored, disable globally with the server's --reasoning-budget 0.
            "chat_template_kwargs": {"enable_thinking": options["think"]},
        }

    def _extract(self, data: dict, options: dict) -> str:
        out = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        if options["strip_think"]:
            out = _THINK.sub("", out)
        return out.strip()

    def _translate(self, model: str, system: str, prompt: str, options: dict) -> str:
        return self._extract(self._post("/v1/chat/completions", self._body(model, system, prompt, options)), options)

    def _translate_batch_call(self, model: str, system: str, prompt: str, schema: dict, options: dict) -> str:
        """Batch: same body plus response_format=json_schema to force the exact
        JSON shape (which also bounds the output length — no max_tokens needed)."""
        body = self._body(model, system, prompt, options)
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "translations", "schema": schema, "strict": True},
        }
        return self._extract(self._post("/v1/chat/completions", body), options)
