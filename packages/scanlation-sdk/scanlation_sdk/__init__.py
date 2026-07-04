"""scanlation-sdk — the single seam shared by the server core and every engine
plugin: the engine contract (EngineBase/Region + role Protocols), the plugin-
facing runtime context (models_dir/device/languages), and the LLM prompt
template. Plugins depend on this, never on the server package."""
from scanlation_sdk.context import LANG_PLAIN, LANGUAGES, context
from scanlation_sdk.contracts import (
    Detector,
    EngineBase,
    Recognizer,
    Region,
    Translator,
)
from scanlation_sdk.device import pick_device, release_cuda_cache
from scanlation_sdk.http_translator import HttpTranslatorBase
from scanlation_sdk.local_engine import LocalModelEngineBase

__all__ = [
    "EngineBase",
    "HttpTranslatorBase",
    "LocalModelEngineBase",
    "Region",
    "Detector",
    "Recognizer",
    "Translator",
    "context",
    "pick_device",
    "release_cuda_cache",
    "LANGUAGES",
    "LANG_PLAIN",
]
