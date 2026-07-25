"""HttpEngineBase — shared lifecycle for engines whose model runs in a SEPARATE
HTTP server (ollama, llama.cpp / any OpenAI-compatible backend).

Every such engine wants the same skeleton regardless of role: a base URL resolved
from an env var at construction, a lazily-created httpx client, one ``_post`` seam
unit tests can monkeypatch, and a model list for the /admin picker. What differs is
the payload — a chat completion for a translator, an image + prompt for a
recognizer — so that stays in the subclass.

Unlike LocalModelEngineBase these hold no VRAM in this process: ``load()`` only
opens a client, so an idle sweep tearing them down costs nothing to rebuild.

httpx is imported lazily inside methods so the SDK's install-time deps stay
numpy + pillow only (the server core imports this module transitively).
"""
from __future__ import annotations

import os

from scanlation_sdk.contracts import EngineBase


def http_timeout() -> float:
    """HTTP client timeout (seconds) for LLM translators — SCANLATION_HTTP_TIMEOUT
    (default 10.0). Read from env directly: the SDK is plugin-facing and does not
    import the server's config."""
    return float(os.getenv("SCANLATION_HTTP_TIMEOUT", "10.0"))


def list_timeout() -> float:
    """HTTP timeout (seconds) for the admin model-list probe —
    SCANLATION_HTTP_LIST_TIMEOUT (default 4.0). Deliberately shorter than
    http_timeout(): the /admin picker must stay responsive, so an unreachable
    backend yields an empty list fast instead of blocking the settings page."""
    return float(os.getenv("SCANLATION_HTTP_LIST_TIMEOUT", "4.0"))


class HttpEngineBase(EngineBase):
    # --- subclass config ---
    ENDPOINT_ENV: str = ""          # env var holding the backend base URL
    DEFAULT_ENDPOINT: str = ""      # fallback base URL when the env var is unset
    ROLE_LABEL: str = "engine"      # only shapes the one-line 'ready' log

    def __init__(self) -> None:
        # rstrip so a trailing slash in the env var can't produce a `//path` URL.
        self.endpoint = os.getenv(self.ENDPOINT_ENV, self.DEFAULT_ENDPOINT).rstrip("/")
        self._client = None

    def _timeout(self) -> float:
        """Per-request budget. Overridden by roles whose calls are slower than a
        translation (a vision recognize runs ~1s/crop and can spike on a cold
        server, so it needs a far longer allowance than the 10s LLM default)."""
        return http_timeout()

    def load(self) -> None:
        if self._client is not None:
            return
        import httpx

        self._client = httpx.Client(timeout=self._timeout())
        self._log.info("%s %s ready (endpoint=%s)", self.name, self.ROLE_LABEL, self.endpoint)

    def unload(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def list_models(self) -> list[str]:
        """Model ids the backend reports (for the admin picker). [] if the backend
        is unreachable or doesn't expose a list — never raises."""
        try:
            import httpx

            resp = httpx.get(self._models_url(), timeout=list_timeout())
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

    # --- subclass hooks ---
    def _models_url(self) -> str:
        raise NotImplementedError

    def _parse_models(self, payload: dict) -> list[str]:
        raise NotImplementedError
