"""Admin endpoints backing the /admin web page.

One read (`GET /get_settings/`) returns everything the page needs; the rest are
small mutations that persist to state.json so the choice survives restarts and
the browser extension no longer has to set models on every page:

  * /set_options/    per-engine option overrides (incl. the LLM model tag)
  * /save_prompt/    upsert a custom system-prompt preset + activate it
  * /select_prompt/  activate an existing preset (builtin or custom)
  * /delete_prompt/  remove a custom preset

Model/lang selection + plugin install reuse the existing wire endpoints
(/set_engines/, /set_languages/, /install_plugins/).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import __version_array__
from scanlation_sdk.context import LANGUAGES
from ..cache import cache
from ..catalog import catalog
from ..engine_meta import class_meta, safe_is_installed, serialize_schema
from ..plugins_install import installing_names
from ..prompts import BUILTIN_PROMPTS
from ..registry import ROLE_NAMES, registry
from ..schemas import (
    SavePromptRequest,
    SelectPromptRequest,
    SetClientConfigRequest,
    SetOptionsRequest,
)
from ..state import state

router = APIRouter()


def _engine_entries(role: str) -> list[dict]:
    """Engines for a role, both installed (pip-present, from the registry) and
    merely installable (source shipped, package not yet pip-installed — from the
    catalog). ``installed_package`` distinguishes the two; ``installed`` = weights.
    Only ``installed_package`` engines are selectable as models (they're in the
    registry); catalog-only ones appear just in the plugin tab to be installed."""
    entries = []
    for name in registry.names(role):
        cls = registry.get_class(role, name)
        entries.append({
            "name": name,
            **class_meta(cls, name),
            "installed": safe_is_installed(cls),
            "installed_package": True,
            "schema": serialize_schema(cls),
            "options": dict(state.selection.options.get(name, {})),
            "device": state.selection.devices.get(name, ""),
        })
    installed_names = {e["name"] for e in entries}
    for name, entry in catalog().items():
        if name in installed_names or role not in entry.roles:
            continue
        entries.append({
            "name": name,
            "display_name": entry.display_name,   # friendly name from the catalog (class not loaded yet)
            "description": entry.description,
            "warning": None,
            "homepage": None,
            "uses_device": False,
            "default_device": "cpu",
            "installed": False,
            "installed_package": False,
            "schema": {},
            "options": {},
            "device": "",
        })
    return entries


@router.get("/get_settings/")
def get_settings() -> dict:
    """Full admin snapshot: selection + engines (w/ schema, install status, saved
    options) + languages + prompt presets. One call drives the whole page."""
    sel = state.selection
    return {
        "version": __version_array__,
        "selection": {
            "detector": sel.detector,
            "recognizer": sel.recognizer,
            "translator": sel.translator,
            "lang_src": sel.lang_src,
            "lang_dst": sel.lang_dst,
            "prompt_active": sel.prompt_active,
            "min_image_dim": sel.min_image_dim,
            "verbose_log": sel.verbose_log,
            "translate_concurrency": sel.translate_concurrency,
        },
        "languages": LANGUAGES,
        "engines": {role: _engine_entries(role) for role in ROLE_NAMES},
        "installing": installing_names(),   # plugins whose install is running now

        "prompts": {
            "active": sel.prompt_active,
            "builtin": BUILTIN_PROMPTS,
            "custom": dict(sel.prompts),
        },
    }


@router.get("/get_translator_models/")
def get_translator_models(engine: str | None = None) -> dict:
    """Model tags installed on a translator's backend (ollama /api/tags,
    llama.cpp /v1/models), so the admin 'model' field can offer a picker.
    Defaults to the active translator; [] if unreachable or not applicable."""
    name = engine or state.selection.translator
    if not registry.has("translator", name):
        return {"models": []}
    try:
        models = registry.get_class("translator", name)().list_models()
    except Exception:  # noqa: BLE001
        models = []
    return {"models": models}


@router.post("/set_options/")
def set_options(req: SetOptionsRequest) -> dict:
    """Persist per-engine option overrides. Engine must exist in some role."""
    known = any(registry.has(role, req.engine) for role in ROLE_NAMES)
    if not known:
        raise HTTPException(status_code=400, detail=f"unknown engine: {req.engine}")
    state.set_options(req.engine, req.options)
    return {"status": "success", "options": dict(state.selection.options.get(req.engine, {}))}


@router.post("/save_prompt/")
def save_prompt(req: SavePromptRequest) -> dict:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="prompt name required")
    state.save_prompt(name, req.text)
    return {"status": "success", "active": state.selection.prompt_active}


@router.post("/select_prompt/")
def select_prompt(req: SelectPromptRequest) -> dict:
    if req.name not in BUILTIN_PROMPTS and req.name not in state.selection.prompts:
        raise HTTPException(status_code=400, detail=f"unknown prompt: {req.name}")
    state.select_prompt(req.name)
    return {"status": "success", "active": state.selection.prompt_active}


@router.post("/delete_prompt/")
def delete_prompt(req: SelectPromptRequest) -> dict:
    if req.name in BUILTIN_PROMPTS:
        raise HTTPException(status_code=400, detail="cannot delete a builtin prompt")
    state.delete_prompt(req.name)
    return {"status": "success", "active": state.selection.prompt_active}


@router.post("/set_client_config/")
def set_client_config(req: SetClientConfigRequest) -> dict:
    """Persist behavior settings (동작 tab): min_image_dim (image filter shorter-side
    px, delivered to the extension via GET /), verbose_log (DEBUG logging toggle,
    re-applied to the live logger), and translate_concurrency (concurrent-translation
    limit, swaps translate_sem at runtime)."""
    if req.min_image_dim is not None and req.min_image_dim < 0:
        raise HTTPException(status_code=400, detail="min_image_dim must be >= 0")
    if req.translate_concurrency is not None and req.translate_concurrency < 1:
        raise HTTPException(status_code=400, detail="translate_concurrency must be >= 1")
    state.set_client_config(
        min_image_dim=req.min_image_dim, verbose_log=req.verbose_log,
        translate_concurrency=req.translate_concurrency,
    )
    return {
        "status": "success",
        "min_image_dim": state.selection.min_image_dim,
        "verbose_log": state.selection.verbose_log,
        "translate_concurrency": state.selection.translate_concurrency,
    }


@router.post("/clear_cache/")
def clear_cache() -> dict:
    """Drop all cached data (page results + translation log) so every page re-runs
    the full pipeline next time."""
    return {"status": "success", "cleared": cache.clear()}
