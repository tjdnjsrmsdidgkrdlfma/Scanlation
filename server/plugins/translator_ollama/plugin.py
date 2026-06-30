"""OllamaTranslator — LLM translation via a local ollama server.

Ported from the user's own tuned config (model_test.py), not the GPLv3 Crivella
plugin: the system prompt + the inline-`system`/`options`/`think:False` request
shape are the user's working setup. Key tunings:
  * think=False  -> ~11x faster on reasoning models (drops hidden <think>)
  * num_ctx=512  -> ~1GiB less KV-cache VRAM (translation inputs are <200 tok)
  * temperature=0, seed=42, top_p=1.0, num_gpu=31  -> deterministic, GPU-resident

ollama runs as a separate service (env OLLAMA_ENDPOINT, default
http://127.0.0.1:11434/api). The HTTP call is isolated in _generate() so the
request-building logic is unit-testable without a live server.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from app.contracts import EngineBase
from plugins.llm_prompt import SYSTEM_PROMPT, build_prompt

logger = logging.getLogger("scanlation.ollama")

DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api"


class OllamaTranslator(EngineBase):
    name = "ollama"
    display_name = "ollama (LLM)"
    homepage = "https://ollama.com"
    description = "LLM translation via a local ollama server (system-prompted, OCR-error tolerant)."
    warning = "Requires a running ollama server (OLLAMA_ENDPOINT) and a pulled model (OLLAMA_MODEL)."
    OPTION_SCHEMA = {
        "num_ctx": {"type": int, "default": 512, "description": "KV-cache context window (translation inputs are short)."},
        "num_gpu": {"type": int, "default": 31, "description": "Layers to offload to GPU."},
        "temperature": {"type": float, "default": 0.0, "description": "Sampling temperature (0 = deterministic)."},
        "seed": {"type": int, "default": 42, "description": "RNG seed."},
        "top_p": {"type": float, "default": 1.0, "description": "Nucleus sampling p."},
        "think": {"type": bool, "default": False, "description": "Enable model 'thinking' (slower; off for speed)."},
    }
    SUPPORTED_SRC: list[str] = []  # any
    SUPPORTED_DST: list[str] = []

    def __init__(self) -> None:
        self.endpoint = os.getenv("OLLAMA_ENDPOINT", DEFAULT_ENDPOINT)
        self.model = os.getenv("OLLAMA_MODEL", "")
        self._client = None

    def load(self) -> None:
        if self._client is not None:
            return
        import httpx

        self._client = httpx.Client(timeout=120.0)
        logger.info("ollama translator ready (endpoint=%s model=%s)", self.endpoint, self.model or "<unset>")

    def unload(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _generate(self, body: dict) -> dict:
        """POST /generate. Isolated so request-building is testable w/o a server."""
        if self._client is None:
            self.load()
        resp = self._client.post(f"{self.endpoint}/generate", json=body)
        resp.raise_for_status()
        return resp.json()

    def translate(self, text: str, src: str, dst: str, options: dict[str, Any]) -> str:
        text = text.strip()
        if len(text) <= 2:  # punctuation/short tokens: not worth a model call
            return text

        options = options or {}
        prompt = build_prompt(text, src, dst, options.get("context", ""))

        body = {
            "model": options.get("model", self.model),
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "think": bool(options.get("think", False)),
            "options": {
                "temperature": float(options.get("temperature", 0.0)),
                "seed": int(options.get("seed", 42)),
                "top_p": float(options.get("top_p", 1.0)),
                "num_gpu": int(options.get("num_gpu", 31)),
                "num_ctx": int(options.get("num_ctx", 512)),
            },
        }
        data = self._generate(body)
        return (data.get("response") or "").strip()
