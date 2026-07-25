"""HttpTranslatorBase — the shared skeleton for LLM translators that talk to a
local HTTP backend (ollama, llama.cpp / any OpenAI-compatible server).

The client lifecycle (endpoint env, lazy httpx client, ``_post``, model list) is
role-agnostic and lives in ``HttpEngineBase``; this adds what is specific to
TRANSLATING: the guardrails (skip blank inputs, require a model chosen in /admin),
the prompt selection (per-call ``system_prompt`` falling back to
``DEFAULT_SYSTEM_PROMPT``), and the batch-with-fallback path. Only two things
differ per backend — the request body shape and how the response text is pulled
out — provided via ``_translate()`` (+ ``_models_url``/``_parse_models``).

``http_timeout``/``list_timeout`` are re-exported here: they moved to
``http_engine`` with the lifecycle, and this stays their established import path.
"""
from __future__ import annotations

import json
from typing import Any

from scanlation_sdk.http_engine import (  # noqa: F401 - re-exported (established import path)
    HttpEngineBase,
    http_timeout,
    list_timeout,
)
from scanlation_sdk.prompt import (
    DEFAULT_SYSTEM_PROMPT,
    batch_schema,
    build_batch_prompt,
    build_prompt,
)

# Sampling options every LLM translator exposes identically. Spread into each
# plugin's OPTION_SCHEMA; `model` stays local (its description differs per
# backend) and so do backend-specific keys (num_ctx/think/strip_think/...).
COMMON_LLM_OPTIONS: dict = {
    "temperature": {"type": float, "default": 0.0, "description": "Sampling temperature (0 = deterministic)."},
    "seed": {"type": int, "default": 42, "description": "RNG seed."},
    "top_p": {"type": float, "default": 1.0, "description": "Nucleus sampling p."},
}


class HttpTranslatorBase(HttpEngineBase):
    # Short timeout (default 10s, SCANLATION_HTTP_TIMEOUT — HttpEngineBase._timeout):
    # think-off translations return in ~1s, so if a request hangs (backend stall) fail
    # over to the per-text fallback fast instead of sitting for minutes. Keep the model
    # warm (OLLAMA_KEEP_ALIVE=-1) so a cold model reload can't eat this budget.
    ROLE_LABEL = "translator"

    def translate(self, text: str, src: str, dst: str, options: dict[str, Any]) -> str:
        text = text.strip()
        if not text:  # blank source: nothing to translate
            return text

        options = self.resolve_options(options)
        model = options.get("model")
        if not model:
            raise ValueError(f"no {self.display_name} model selected — pick one in /admin")
        system = options.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        prompt = build_prompt(text, src, dst, options.get("context", ""))
        return self._translate(model, system, prompt, options)

    def translate_batch(
        self, texts: list[str], src: str, dst: str, options: dict[str, Any]
    ) -> list[str]:
        """Translate many texts in one model call and return them aligned to the
        input order. Blank texts pass through unchanged; every non-blank text goes
        to the model. ANY failure (parse error, wrong count, HTTP error, num_ctx
        overflow -> truncated JSON) falls back to a per-text translate() loop, so
        the result is always complete and aligned — just slower on that page."""
        options = self.resolve_options(options)
        stripped = [t.strip() for t in texts]
        idx = [i for i, t in enumerate(stripped) if t]  # non-blank -> worth a model call
        if not idx:  # all blank
            return stripped

        model = options.get("model")
        if not model:
            raise ValueError(f"no {self.display_name} model selected — pick one in /admin")
        items = [stripped[i] for i in idx]
        system = options.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        prompt = build_batch_prompt(items, src, dst, options.get("context", ""))
        schema = batch_schema(len(items))
        raw = None
        try:
            raw = self._translate_batch_call(model, system, prompt, schema, options)
            obj = json.loads(raw)
            translated = [obj[f"t{i}"] for i in range(len(items))]  # KeyError -> fallback
        except Exception as e:  # noqa: BLE001 - any failure -> safe per-text fallback
            # Surface WHY the batch failed so it's diagnosable from logs: JSONDecodeError
            # = truncated JSON (num_ctx too small), HTTPStatusError = backend, KeyError =
            # wrong item count. The raw response (DEBUG) pins truncation vs bad shape.
            self._log.warning(
                "%s batch of %d failed (%s: %s); falling back to per-text",
                self.name, len(items), type(e).__name__, e,
            )
            self._log.debug("batch prompt=%d chars; raw response=%r",
                            len(prompt), (raw[:500] if isinstance(raw, str) else raw))
            translated = [self.translate(t, src, dst, options) for t in items]

        out = list(stripped)  # blanks kept in place
        for i, tr in zip(idx, translated):
            out[i] = (tr if isinstance(tr, str) else str(tr)).strip()
        return out

    # --- subclass hooks ---
    def _translate(self, model: str, system: str, prompt: str, options: dict) -> str:
        """Build the backend request, send it, and return the translated text."""
        raise NotImplementedError

    def _translate_batch_call(
        self, model: str, system: str, prompt: str, schema: dict, options: dict
    ) -> str:
        """Send a batch request whose output is constrained to ``schema`` (native
        structured output) and return the raw JSON string. Subclass hook, mirrors
        ``_translate``."""
        raise NotImplementedError
