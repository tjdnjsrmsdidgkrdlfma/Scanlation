"""The builtin system prompt + preset resolution.

The active translation system prompt is part of the server's persisted selection
(state.json), chosen/edited from the admin page. Translator plugins receive the
resolved text through the per-call options dict (``system_prompt``), so they stay
decoupled from this module — and fall back to ``DEFAULT_SYSTEM_PROMPT`` when
called directly (unit tests / a bare ``translate()``).

The only builtin preset is ``default``; user-saved custom presets
(state.selection.prompts) layer on top. The baseline ``DEFAULT_SYSTEM_PROMPT``
and the user-turn ``build_prompt`` template live in ``scanlation_sdk.prompt``,
shared with the translator plugins.
"""
from __future__ import annotations

# The baseline default prompt lives in the SDK (shared with the translator
# plugins); this module registers it as the sole builtin and lets user custom
# presets (from state) layer on top.
from scanlation_sdk.prompt import DEFAULT_SYSTEM_PROMPT

# name -> text. Only "default" is builtin; the admin dropdown lists it plus any
# user-saved custom prompts (state.selection.prompts).
BUILTIN_PROMPTS: dict[str, str] = {"default": DEFAULT_SYSTEM_PROMPT}


def resolve_prompt(active: str, custom: dict[str, str] | None) -> str:
    """The system-prompt text for the active preset name.

    User-saved ``custom`` presets (from state) take precedence over the builtins
    of the same name; an unknown name falls back to the default.
    """
    custom = custom or {}
    if active in custom:
        return custom[active]
    if active in BUILTIN_PROMPTS:
        return BUILTIN_PROMPTS[active]
    return DEFAULT_SYSTEM_PROMPT
