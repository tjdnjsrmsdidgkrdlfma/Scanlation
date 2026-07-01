"""Pydantic wire models shared by the server and the bundled MV3 extension.

Engine roles are named detector/recognizer/translator end-to-end (the old
ocr_extension BOX/OCR/TSL vocabulary was dropped). Per-result item keys
(ocr/tsl/box) are data fields, not roles, and stay as-is.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


# --- /run_ocrtsl/ ----------------------------------------------------------
class RunOcrTslRequest(BaseModel):
    md5: str
    contents: Optional[str] = None          # base64; absent => lazy cache lookup
    options: Optional[dict[str, Any]] = None  # {engine_name: {opt: val}}
    force: Optional[bool] = False           # extension never sends this; re-run + overwrite


class ResultItem(BaseModel):
    ocr: str
    tsl: str
    box: list[int]                          # [x_min, y_min, x_max, y_max] == client [l,b,r,t]


class RunOcrTslResponse(BaseModel):
    result: list[ResultItem]


# --- /run_tsl/ -------------------------------------------------------------
class RunTslRequest(BaseModel):
    text: str


class RunTslResponse(BaseModel):
    text: str


# --- /set_manual_translation/ ---------------------------------------------
class SetManualRequest(BaseModel):
    text: str
    translation: str


# --- /set_models/ ----------------------------------------------------------
class SetModelsRequest(BaseModel):
    detector: Optional[str] = None
    recognizer: Optional[str] = None
    translator: Optional[str] = None


# --- /set_lang/ ------------------------------------------------------------
class SetLangRequest(BaseModel):
    lang_src: str
    lang_dst: str


# --- /manage_plugins/ ------------------------------------------------------
class ManagePluginsRequest(BaseModel):
    plugins: dict[str, bool]


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
