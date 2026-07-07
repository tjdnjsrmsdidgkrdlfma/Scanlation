"""Pydantic wire models shared by the server and the bundled MV2 extension.

Engine roles are named detector/recognizer/translator end-to-end (the old
ocr_extension BOX/OCR/TSL vocabulary was dropped). Per-result item keys are
``{bounds, source, destination}`` — data fields, not roles. Only request bodies
are modeled — responses are plain dicts (the wire shapes live in the routes).
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


# --- /run_pipeline/ + /run_lookup/ (shared request body) --------------------
class RunRequest(BaseModel):
    md5: str
    contents: Optional[str] = None          # base64; required by /run_pipeline/, ignored by /run_lookup/
    options: Optional[dict[str, Any]] = None  # {engine_name: {opt: val}}
    force: Optional[bool] = False           # extension never sends this; re-run + overwrite


# --- /set_engines/ ----------------------------------------------------------
class SetEnginesRequest(BaseModel):
    detector: Optional[str] = None
    recognizer: Optional[str] = None
    translator: Optional[str] = None


# --- /set_languages/ ------------------------------------------------------------
class SetLanguagesRequest(BaseModel):
    lang_src: str
    lang_dst: str


# --- /set_engine_device/ ----------------------------------------------------
class SetEngineDeviceRequest(BaseModel):
    engine: str
    device: str = ""                     # "cpu"/"cuda"/"cuda:N"; "" removes the override -> DEFAULT_DEVICE


# --- /install_plugins/ ------------------------------------------------------
class InstallPluginsRequest(BaseModel):
    plugins: dict[str, bool]


# --- /install_plugin_stream/ (one plugin, live NDJSON progress) -------------
class InstallPluginStreamRequest(BaseModel):
    name: str


# --- admin: /set_options/ --------------------------------------------------
class SetOptionsRequest(BaseModel):
    engine: str
    options: dict[str, Any]              # {opt: val}; null/"" removes the override


# --- admin: prompt presets -------------------------------------------------
class SavePromptRequest(BaseModel):
    name: str
    text: str


class SelectPromptRequest(BaseModel):
    name: str


# --- admin: /set_client_config/ (동작 tab) ---------------------------------
class SetClientConfigRequest(BaseModel):
    min_image_dim: Optional[int] = None   # extension image filter (shorter-side px)
    verbose_log: Optional[bool] = None    # DEBUG logging toggle (per-detection/translation detail)
    translate_concurrency: Optional[int] = None  # max images translating at once (swaps translate_sem)
