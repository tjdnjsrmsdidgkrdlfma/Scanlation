# Scanlation extension (MV2 · Firefox)

Clean-room Manifest V2 browser extension — no bundler, no npm, plain ES.
Overlays the Scanlation server's translations in place over manga images.

Firefox-only: MV2 gives two toolbar surfaces like the original ocr_extension —
a settings **popup** (`browser_action`) and an address-bar **toggle icon**
(`page_action`). Chrome has removed MV2, so this loads in Firefox.

Wire contract matches the server exactly (verified end-to-end): md5 over the
**base64 string**, box `[x_min, y_min, x_max, y_max]`, `/run_pipeline/` lazy(md5)
→ work(contents) on a cache miss. The md5 implementation is byte-equal to
Python's `hashlib.md5`.

## Load (Firefox)

`about:debugging#/runtime/this-firefox` → *Load Temporary Add-on* → pick
`extension/manifest.json`.

## Use

- **Settings** — click the **toolbar icon** (`browser_action`) → the popup opens.
  Confirm the **Server** URL → *Connect* (dropdowns fill from the handshake),
  pick languages / engines. (Model/lang can also be set server-side at `/admin`.)
- **Translate** — click the **address-bar icon** (`page_action`) to toggle
  translation on the tab. **One click.** The icon shows state: grey = off,
  colour = on. Click again to turn off. The popup's *Enable / Disable* buttons
  do the same (the address-bar icon stays in sync). Click a box to copy the
  original text; *Show translated* flips translated / original.

## Files

```
manifest.json          MV2 manifest (browser_action popup + page_action toggle)
content.css            overlay box styles
popup.html/css         settings popup (dark theme)
icons/                 icon{16,48,128} (on) + icon-off{16,48} (page_action off state)
src/content.js         image discovery + lazy/work + overlay
src/background.js      page_action toggle + icon sync + cross-origin image fetch
src/popup.js           handshake-driven settings controls
src/starfield.js       popup background animation
src/constants.js       shared config defaults on globalThis.SCAN (endpoint, min image dim)
src/util.js            shared pure helpers (endpoint/auth/base64/box math) on globalThis.SCANUTIL
src/i18n.js            shared en/ko strings (popup + content failure badge) on globalThis.SCANI18N
src/md5.js             clean-room md5 (byte-equal to Python hashlib.md5)
```

No build step: the files load as-is. Manifest content scripts and the event page
can't use ES imports, so the shared pieces (`constants.js`, `util.js`, `i18n.js`,
`md5.js`) are classic scripts that publish onto `globalThis` and are listed before
their consumers in each load list. The `icon-off*` PNGs are generated from our own
`icon*` (desaturated + dimmed) — no GPLv3 assets from ocr_extension are copied,
keeping the tree clean-room.
