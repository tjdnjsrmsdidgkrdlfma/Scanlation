"""Shared LLM-translation prompt template, used by every LLM-backed translator
(ollama, llama.cpp, any OpenAI-compatible server) so the user-turn shape stays
consistent across backends.

The *system* prompt is no longer hardcoded here: it's chosen/edited from the
admin page and flows to translators through the per-call options dict
(``system_prompt``). ``SYSTEM_PROMPT`` is re-exported from ``app.prompts`` only
as the fallback default for a bare ``translate()`` call (unit tests).
"""
from __future__ import annotations

from app.config import LANG_PLAIN
from app.prompts import DEFAULT_SYSTEM_PROMPT as SYSTEM_PROMPT  # re-export (fallback default)

__all__ = ["SYSTEM_PROMPT", "build_prompt"]


def build_prompt(text: str, src: str, dst: str, context: str = "") -> str:
    """The user-turn prompt. src/dst are mapped to plain names (ja -> japanese)."""
    s = LANG_PLAIN.get(src, src)
    d = LANG_PLAIN.get(dst, dst)
    return f'src="{s}"\ndst="{d}"\ncontext="{context}"\ntext="{text}"'
