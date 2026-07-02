"""OllamaTranslator — LLM translation via a local ollama server.

Ported from the user's own tuned config (model_test.py), not the GPLv3 Crivella
plugin: the system prompt + the inline-`system`/`options`/`think:False` request
shape are the user's working setup. Key tunings:
  * think=False  -> ~11x faster on reasoning models (drops hidden <think>)
  * num_ctx=512  -> ~1GiB less KV-cache VRAM (translation inputs are <200 tok)
  * temperature=0, seed=42, top_p=1.0, num_gpu=31  -> deterministic, GPU-resident

ollama runs as a separate service (env OLLAMA_ENDPOINT, default
http://127.0.0.1:11434/api). The client lifecycle + guardrails live in
HttpTranslatorBase; only the /generate body shape + response field are here.
The HTTP call is isolated in _generate() so request-building is unit-testable.
"""
from __future__ import annotations

from scanlation_sdk.http_translator import HttpTranslatorBase


class OllamaTranslator(HttpTranslatorBase):
    name = "ollama"
    display_name = "Ollama"
    homepage = "https://ollama.com"
    description = "LLM translation via a local ollama server (system-prompted, OCR-error tolerant)."
    warning = "Requires a running ollama server (OLLAMA_ENDPOINT) and a model pulled + selected in /admin."
    ENDPOINT_ENV = "OLLAMA_ENDPOINT"
    DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api"
    OPTION_SCHEMA = {
        "model": {"type": str, "default": "", "description": "ollama model tag (e.g. gemma4:31b). Required — pick it in /admin."},
        "num_ctx": {"type": int, "default": 512, "description": "KV-cache context window (translation inputs are short)."},
        "num_gpu": {"type": int, "default": 31, "description": "Layers to offload to GPU."},
        "temperature": {"type": float, "default": 0.0, "description": "Sampling temperature (0 = deterministic)."},
        "seed": {"type": int, "default": 42, "description": "RNG seed."},
        "top_p": {"type": float, "default": 1.0, "description": "Nucleus sampling p."},
        "think": {"type": bool, "default": False, "description": "Enable model 'thinking' (slower; off for speed)."},
    }

    def _generate(self, body: dict) -> dict:
        """POST /generate. Kept as the unit-test seam (tests fake this)."""
        return self._post("/generate", body)

    def _models_url(self) -> str:
        return f"{self.endpoint}/tags"

    def _parse_models(self, payload: dict) -> list[str]:
        return [m["name"] for m in payload.get("models", []) if m.get("name")]

    def _translate(self, model: str, system: str, prompt: str, options: dict) -> str:
        body = {
            "model": model,
            "prompt": prompt,
            "system": system,
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
