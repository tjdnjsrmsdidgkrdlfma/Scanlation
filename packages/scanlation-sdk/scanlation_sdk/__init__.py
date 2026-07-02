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
from scanlation_sdk.http_translator import HttpTranslatorBase

__all__ = [
    "EngineBase",
    "HttpTranslatorBase",
    "Region",
    "Detector",
    "Recognizer",
    "Translator",
    "context",
    "LANGUAGES",
    "LANG_PLAIN",
]
