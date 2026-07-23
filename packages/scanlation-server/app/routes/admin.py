"""Admin endpoints backing the /admin web page.

One read (`GET /get_settings/`) returns everything the page needs; the rest are
small mutations that persist to state.json so the choice survives restarts and
the browser extension no longer has to set models on every page. They cover
per-engine option overrides (incl. the LLM model tag), custom system-prompt
presets (save/select/delete), the extension client config, a translator
backend's available-models query, and cache/stats maintenance — plus the
recognize-pool occupancy bench readouts (a diagnostic aid).

Model/lang selection + plugin install reuse the existing wire endpoints
(/set_engines/, /set_languages/, /install_plugins/).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import __version_array__
from scanlation_sdk.context import LANGUAGES
from . import require_known_engine
from ..cache import cache
from ..config import settings
from ..catalog import catalog
from ..engine_meta import class_meta, safe_is_installed, serialize_schema
from ..gpus import detect_gpu_vendor, installed_torch_build, list_gpus
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
            # per-engine recognize worker-pool override ("" = the global default);
            # the admin UI shows this field only for recognizers that load onto a device
            "recognize_concurrency": state.selection.recognize_concurrency.get(name, ""),
            # per-recognizer gate size (max concurrent images); same UI condition as above
            "gpu_concurrency": state.selection.gpu_concurrency.get(name, ""),
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
            "recognize_concurrency": "",
            "gpu_concurrency": "",
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
            "model_idle_unload_minutes": sel.model_idle_unload_minutes,
            "torch_backend": sel.torch_backend,
            "torch_vendor": sel.torch_vendor,
            "torch_index": sel.torch_index,
        },
        "languages": LANGUAGES,
        # Global fallback the per-engine recognize-worker field shows as its placeholder
        # (an engine with no override runs this many workers; 1 = no pool).
        "recognize_concurrency_default": settings.recognize_concurrency,
        # Global fallback for the per-recognizer gate-size field (1 = serial images).
        "gpu_concurrency_default": settings.gpu_concurrency,
        "gpus": list_gpus(),                # [{index, name}] for the per-engine device picker
        "gpu_vendor": detect_gpu_vendor(),  # amd/nvidia/both/None from device nodes (torch backend auto-pick)
        "torch_build": installed_torch_build(),  # cpu/cuda/rocm/None — for the backend-mismatch warning
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
    require_known_engine(req.engine)
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
    limit, swaps translate_sem at runtime) plus model_idle_unload_minutes (idle
    minutes before a local engine leaves VRAM; read live by the background sweep).
    Out-of-range values are clamped by state.set_client_config (min_image_dim >= 0,
    translate_concurrency >= 1, model_idle_unload_minutes >= 0), the single
    validation authority — so the route trusts its input."""
    state.set_client_config(
        min_image_dim=req.min_image_dim, verbose_log=req.verbose_log,
        translate_concurrency=req.translate_concurrency,
        model_idle_unload_minutes=req.model_idle_unload_minutes,
        torch_backend=req.torch_backend, torch_vendor=req.torch_vendor,
        torch_index=req.torch_index,
    )
    return {
        "status": "success",
        "min_image_dim": state.selection.min_image_dim,
        "verbose_log": state.selection.verbose_log,
        "translate_concurrency": state.selection.translate_concurrency,
        "model_idle_unload_minutes": state.selection.model_idle_unload_minutes,
        "torch_backend": state.selection.torch_backend,
        "torch_vendor": state.selection.torch_vendor,
        "torch_index": state.selection.torch_index,
    }


@router.post("/clear_cache/")
def clear_cache() -> dict:
    """Drop all cached data (page results + translation log) so every page re-runs
    the full pipeline next time."""
    return {"status": "success", "cleared": cache.clear()}


@router.get("/get_stats/")
def get_stats(engines: str | None = None) -> dict:
    """Per-page + per-crop processing stats: count + mean/min/max/median/p90/p99 per
    numeric column, for the 통계 tab. Benchmark (skip_translate) pages are excluded.
    Optional ``?engines=`` filters to one pipeline config."""
    return cache.stats_summary(engines)


@router.post("/clear_stats/")
def clear_stats() -> dict:
    """Drop all processing-stats history (both tables). Separate from /clear_cache/ —
    clearing the page cache means 'recompute', not 'forget the stats'."""
    return {"status": "success", "cleared": cache.clear_stats()}


# --- TEMP occupancy bench (revert via git) -----------------------------------
# Measures how full the recognize worker pool was during a batch, to verify the
# "gate+K is an image bundle -> workers idle when crops sum < W" claim directly.
# Workflow: POST /bench_occupancy_reset/ -> fire a --parallel --no-translate batch ->
# GET /bench_occupancy/ (see tools/bench_occupancy.py, which does all three per K).
@router.post("/bench_occupancy_reset/")
def bench_occupancy_reset() -> dict:
    from ..recognize_pool import reset_occupancy

    reset_occupancy()
    return {"status": "success"}


@router.get("/bench_occupancy/")
def bench_occupancy() -> dict:
    from ..recognize_pool import active_workers, occupancy_stats

    return occupancy_stats(active_workers())
