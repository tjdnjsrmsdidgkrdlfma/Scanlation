"""Backward-compat shim: the shared LLM prompt template moved to the SDK
(``scanlation_sdk.prompt``). Kept only while the ollama/llamacpp plugins still
live in-tree; it is removed once they become standalone packages that import the
SDK directly.
"""
from scanlation_sdk.prompt import SYSTEM_PROMPT, build_prompt

__all__ = ["SYSTEM_PROMPT", "build_prompt"]
