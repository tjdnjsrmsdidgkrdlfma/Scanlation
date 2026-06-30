"""LlamaCppTranslator — translation via an OpenAI-compatible chat endpoint.

Primary use: run the LLM under llama.cpp's `llama-server` with the **Vulkan**
backend, which is often more reliable than ROCm on newer AMD GPUs (e.g. gfx1200
/ RDNA4) where ollama's Vulkan support is weak. Same system prompt + template as
the ollama backend (shared via plugins.llm_prompt), so translations are
consistent regardless of which backend the GPU happens to like.

Talks `/v1/chat/completions`, so it also works with any OpenAI-compatible server
(vllm, LM Studio, ollama's own /v1, etc.). Reasoning-model `<think>...</think>`
spans are stripped (llama.cpp has no ollama-style think:false toggle).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from app.contracts import EngineBase
from plugins.llm_prompt import SYSTEM_PROMPT, build_prompt

logger = logging.getLogger("scanlation.llamacpp")

DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LlamaCppTranslator(EngineBase):
    name = "llamacpp"
    display_name = "llama.cpp / OpenAI-compatible"
    homepage = "https://github.com/ggml-org/llama.cpp"
    description = "Translation via an OpenAI-compatible /v1/chat/completions server (llama.cpp Vulkan, vllm, LM Studio...)."
    warning = "Requires a running server (LLAMACPP_ENDPOINT, default http://127.0.0.1:8080) with a model loaded."
    OPTION_SCHEMA = {
        "model": {"type": str, "default": "", "description": "Model name (OpenAI clients require non-empty; llama-server ignores). Blank = LLAMACPP_MODEL env."},
        "temperature": {"type": float, "default": 0.0, "description": "Sampling temperature (0 = deterministic)."},
        "seed": {"type": int, "default": 42, "description": "RNG seed."},
        "top_p": {"type": float, "default": 1.0, "description": "Nucleus sampling p."},
        "max_tokens": {"type": int, "default": 512, "description": "Max tokens to generate."},
        "strip_think": {"type": bool, "default": True, "description": "Remove <think>...</think> from reasoning models."},
    }
    SUPPORTED_SRC: list[str] = []
    SUPPORTED_DST: list[str] = []

    def __init__(self) -> None:
        self.endpoint = os.getenv("LLAMACPP_ENDPOINT", DEFAULT_ENDPOINT)
        self.model = os.getenv("LLAMACPP_MODEL", "local")  # llama-server ignores this, OpenAI clients require it
        self._client = None

    def load(self) -> None:
        if self._client is not None:
            return
        import httpx

        self._client = httpx.Client(timeout=120.0)
        logger.info("llama.cpp translator ready (endpoint=%s model=%s)", self.endpoint, self.model)

    def unload(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _chat(self, body: dict) -> dict:
        """POST /v1/chat/completions. Isolated so request-building is testable."""
        if self._client is None:
            self.load()
        resp = self._client.post(f"{self.endpoint.rstrip('/')}/v1/chat/completions", json=body)
        resp.raise_for_status()
        return resp.json()

    def translate(self, text: str, src: str, dst: str, options: dict[str, Any]) -> str:
        text = text.strip()
        if len(text) <= 2:
            return text

        options = options or {}
        body = {
            "model": options.get("model") or self.model,
            "messages": [
                {"role": "system", "content": options.get("system_prompt") or SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(text, src, dst, options.get("context", ""))},
            ],
            "temperature": float(options.get("temperature", 0.0)),
            "top_p": float(options.get("top_p", 1.0)),
            "seed": int(options.get("seed", 42)),
            "max_tokens": int(options.get("max_tokens", 512)),
            "stream": False,
        }
        data = self._chat(body)
        out = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        if options.get("strip_think", True):
            out = _THINK.sub("", out)
        return out.strip()
