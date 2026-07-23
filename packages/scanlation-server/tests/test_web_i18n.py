"""Guard: the admin SPA's i18n table (app/web/i18n.js) keeps en and ko in sync.

The extension has this guard in JS (extension/tests/run.mjs); the larger admin
table had none, so a UI-chrome key added to one language but not the other would
silently fall back (now to English — see t()). No JS runtime here, so this parses
the two language object literals out of the classic script and compares key sets.

The ``opt.<engine>.<key>`` / ``opt.<key>`` namespace is exempt from parity: it is a
ko-only override for engine option descriptions — en falls straight through to the
server's English schema string (see optDesc in app.js). So chrome keys must match
across languages, and en must carry no ``opt.*`` overrides at all.
"""
from __future__ import annotations

import re
from pathlib import Path

I18N_JS = Path(__file__).resolve().parents[1] / "app" / "web" / "i18n.js"


def _lang_keys(text: str, lang: str) -> set[str]:
    """Message keys in the ``<lang>: { ... }`` object literal — the quoted token
    before the colon at the start of each entry line (one entry per line)."""
    m = re.search(rf"^  {lang}: \{{\s*$(.*?)^  \}}", text, re.S | re.M)
    assert m, f"{lang} block not found in {I18N_JS.name}"
    return set(re.findall(r'^\s*"([^"]+)":', m.group(1), re.M))


def test_admin_i18n_key_parity():
    text = I18N_JS.read_text(encoding="utf-8")
    ko = _lang_keys(text, "ko")
    en = _lang_keys(text, "en")
    assert ko, "no ko keys parsed"
    assert en, "no en keys parsed"
    # opt.* is a ko-only override namespace (en falls through to the server string).
    assert not {k for k in en if k.startswith("opt.")}, \
        "en must carry no opt.* overrides (the server description is already English)"
    ko_chrome = {k for k in ko if not k.startswith("opt.")}
    en_chrome = {k for k in en if not k.startswith("opt.")}
    assert ko_chrome == en_chrome, \
        {"only_ko": sorted(ko_chrome - en_chrome), "only_en": sorted(en_chrome - ko_chrome)}


TESTS = [test_admin_i18n_key_parity]

if __name__ == "__main__":
    import sys

    from scanlation_sdk.testing import run

    sys.exit(run(TESTS, "test_web_i18n"))
