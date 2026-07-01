"""LLM system-prompt presets + resolution.

The active translation system prompt is part of the server's persisted selection
(state.json), chosen/edited from the admin page. Translator plugins receive the
resolved text through the per-call options dict (``system_prompt``), so they stay
decoupled from this module — and fall back to ``DEFAULT_SYSTEM_PROMPT`` when
called directly (unit tests / a bare ``translate()``).

The named presets live here (core); the baseline ``DEFAULT_SYSTEM_PROMPT`` and
the user-turn ``build_prompt`` template are in ``scanlation_sdk.prompt``, shared
with the translator plugins.
"""
from __future__ import annotations

# The baseline default prompt lives in the SDK (shared with the translator
# plugins); this module layers the named presets + user custom presets on top.
from scanlation_sdk.prompt import DEFAULT_SYSTEM_PROMPT

# A faithful/literal variant: stay close to the source, preserve honorifics and
# proper nouns, don't localize idioms.
LITERAL_SYSTEM_PROMPT = (
    "You are a precise manga translator. Translate the given text and output only the translation.\n"
    'Input fields: src="source language", dst="target language", '
    'context="optional scene context", text="text to translate".\n'
    "Translate faithfully and literally. Preserve honorifics (e.g. -san, -chan), "
    "names, and onomatopoeia romanization when no natural equivalent exists.\n"
    "Tolerate OCR errors: infer the intended characters from context.\n"
    "Do not add notes, explanations, or quotation marks. Output the translation immediately."
)

# A localization-first variant: natural target-language phrasing, manga register,
# adapt SFX and idioms.
NATURAL_SYSTEM_PROMPT = (
    "You are a professional manga localizer. Translate the given text into natural, "
    "fluent target-language dialogue and output only the translation.\n"
    'Input fields: src="source language", dst="target language", '
    'context="optional scene context", text="text to translate".\n'
    "Prefer how a native speaker would actually say it over a literal rendering. "
    "Adapt idioms and sound effects to natural equivalents. Match the speaker's tone "
    "(casual, rude, cute, formal) to the scene.\n"
    "Tolerate OCR errors and use the context to disambiguate. "
    "No notes or quotation marks — just the localized line."
)

# name -> text. "default" must always exist; the admin dropdown lists these plus
# any user-saved custom prompts (state.selection.prompts).
BUILTIN_PROMPTS: dict[str, str] = {
    "default": DEFAULT_SYSTEM_PROMPT,
    "literal": LITERAL_SYSTEM_PROMPT,
    "natural": NATURAL_SYSTEM_PROMPT,
}


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
