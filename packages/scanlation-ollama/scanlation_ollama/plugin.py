"""OllamaTranslator — LLM translation via a local ollama server.

Ported from the user's own tuned config (model_test.py), not the GPLv3 Crivella
plugin: the system prompt + the inline-`system`/`options`/`think:False` request
shape are the user's working setup. Key tunings:
  * think=False  -> ~11x faster on reasoning models (drops hidden <think>)
  * num_ctx=2048 -> one KV size for the single + batch paths (no model reload
                    when the pipeline switches between them)
  * temperature=0, seed=42, top_p=1.0, num_gpu=31  -> deterministic, GPU-resident

ollama runs as a separate service (env OLLAMA_ENDPOINT, default
http://127.0.0.1:11434/api). The client lifecycle + guardrails live in
HttpTranslatorBase; only the /generate body shape + response field are here.
The HTTP call is isolated in _generate() so request-building is unit-testable.
"""
from __future__ import annotations

from scanlation_sdk.http_translator import HttpTranslatorBase


class OllamaTranslator(HttpTranslatorBase):
    name = "Ollama"
    display_name = "Ollama"
    homepage = "https://ollama.com"
    description = "LLM translation via a local ollama server (must be running, model selected in /admin)."
    ENDPOINT_ENV = "OLLAMA_ENDPOINT"
    DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api"
    OPTION_SCHEMA = {
        "model": {"type": str, "default": "", "description": "ollama model tag (e.g. gemma4:31b). Required — pick it in /admin."},
        "num_ctx": {"type": int, "default": 2048, "description": "KV-cache context window (holds a whole image's batch + its translations)."},
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

    def _sampling(self, options: dict) -> dict:
        """The shared ollama `options` sub-dict. Options arrive already resolved
        against OPTION_SCHEMA (defaults filled + typed) by resolve_options, so this
        just reads them. num_ctx is one value for the single + batch paths, so
        ollama never reloads the model when the pipeline switches between them."""
        return {
            "temperature": options["temperature"],
            "seed": options["seed"],
            "top_p": options["top_p"],
            "num_gpu": options["num_gpu"],
            "num_ctx": options["num_ctx"],
        }

    def _translate(self, model: str, system: str, prompt: str, options: dict) -> str:
        body = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "think": options["think"],
            "options": self._sampling(options),
        }
        data = self._generate(body)
        return (data.get("response") or "").strip()

    def _translate_batch_call(self, model: str, system: str, prompt: str, schema: dict, options: dict) -> str:
        """Batch: same body as _translate plus `format`=schema to force JSON. Uses
        the same num_ctx as _translate so switching between the two never reloads."""
        body = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "think": options["think"],
            "format": schema,
            "options": self._sampling(options),
        }
        return (self._generate(body).get("response") or "").strip()
