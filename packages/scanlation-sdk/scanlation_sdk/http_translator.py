"""HttpTranslatorBase — the shared skeleton for LLM translators that talk to a
local HTTP backend (ollama, llama.cpp / any OpenAI-compatible server).

Both backends want the exact same lifecycle (a lazily-created httpx client keyed
off an endpoint env var), the same guardrails (skip ≤2-char inputs, require a
model chosen in /admin), and the same prompt selection (per-call ``system_prompt``
falling back to ``DEFAULT_SYSTEM_PROMPT``). Only two things actually differ: the
request body shape and how the response text is pulled out. Subclasses provide
those via ``_translate()`` (+ ``_models_url``/``_parse_models`` for the picker).

httpx is imported lazily inside methods so the SDK's install-time deps stay
numpy + pillow only (the server core imports this module transitively).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from scanlation_sdk.contracts import EngineBase
from scanlation_sdk.prompt import (
    DEFAULT_SYSTEM_PROMPT,
    batch_schema,
    build_batch_prompt,
    build_prompt,
)


class HttpTranslatorBase(EngineBase):
    # --- subclass config ---
    ENDPOINT_ENV: str = ""          # env var holding the backend base URL
    DEFAULT_ENDPOINT: str = ""      # fallback base URL when the env var is unset

    def __init__(self) -> None:
        # rstrip so a trailing slash in the env var can't produce a `//path` URL.
        self.endpoint = os.getenv(self.ENDPOINT_ENV, self.DEFAULT_ENDPOINT).rstrip("/")
        self._client = None

    @property
    def _log(self) -> logging.Logger:
        return logging.getLogger(f"scanlation.{self.name}")

    def load(self) -> None:
        if self._client is not None:
            return
        import httpx

        # Short timeout: think-off translations return in ~1s, so if a request
        # hangs (backend stall) fail over to the per-text fallback fast instead of
        # sitting for minutes. Keep the model warm (OLLAMA_KEEP_ALIVE=-1) so a cold
        # 14GB reload can't eat this budget.
        self._client = httpx.Client(timeout=10.0)
        self._log.info("%s translator ready (endpoint=%s)", self.name, self.endpoint)

    def unload(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def list_models(self) -> list[str]:
        """Model ids the backend reports (for the admin picker). [] if the backend
        is unreachable or doesn't expose a list — never raises."""
        try:
            import httpx

            resp = httpx.get(self._models_url(), timeout=4.0)
            resp.raise_for_status()
            return sorted(self._parse_models(resp.json()))
        except Exception:  # noqa: BLE001 - backend unreachable is expected; picker just stays empty
            return []

    def _post(self, path: str, body: dict) -> dict:
        """POST ``body`` to ``endpoint + path`` and return the parsed JSON.
        Isolated so request-building stays unit-testable without a live server."""
        if self._client is None:
            self.load()
        resp = self._client.post(f"{self.endpoint}{path}", json=body)
        resp.raise_for_status()
        return resp.json()

    def translate(self, text: str, src: str, dst: str, options: dict[str, Any]) -> str:
        text = text.strip()
        if len(text) <= 2:  # punctuation/short tokens: not worth a model call
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
        input order. Short (<=2 char) texts are passed through unchanged, same as
        translate(). ANY failure (parse error, wrong count, HTTP error, num_ctx
        overflow -> truncated JSON) falls back to a per-text translate() loop, so
        the result is always complete and aligned — just slower on that page."""
        options = self.resolve_options(options)
        stripped = [t.strip() for t in texts]
        long_idx = [i for i, t in enumerate(stripped) if len(t) > 2]
        if not long_idx:  # nothing worth a model call (all short/empty)
            return stripped

        model = options.get("model")
        if not model:
            raise ValueError(f"no {self.display_name} model selected — pick one in /admin")
        longs = [stripped[i] for i in long_idx]
        system = options.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        prompt = build_batch_prompt(longs, src, dst, options.get("context", ""))
        schema = batch_schema(len(longs))
        raw = None
        try:
            raw = self._translate_batch_call(model, system, prompt, schema, options)
            obj = json.loads(raw)
            translated = [obj[f"t{i}"] for i in range(len(longs))]  # KeyError -> fallback
        except Exception as e:  # noqa: BLE001 - any failure -> safe per-text fallback
            # Surface WHY the batch failed so it's diagnosable from logs: JSONDecodeError
            # = truncated JSON (num_ctx too small), HTTPStatusError = backend, KeyError =
            # wrong item count. The raw response (DEBUG) pins truncation vs bad shape.
            self._log.warning(
                "%s batch of %d failed (%s: %s); falling back to per-text",
                self.name, len(longs), type(e).__name__, e,
            )
            self._log.debug("batch prompt=%d chars; raw response=%r",
                            len(prompt), (raw[:500] if isinstance(raw, str) else raw))
            translated = [self.translate(t, src, dst, options) for t in longs]

        out = list(stripped)  # start from the passthrough (short texts kept in place)
        for i, tr in zip(long_idx, translated):
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

    def _models_url(self) -> str:
        raise NotImplementedError

    def _parse_models(self, payload: dict) -> list[str]:
        raise NotImplementedError
