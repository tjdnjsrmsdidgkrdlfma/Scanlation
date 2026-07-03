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
src/content.js         image discovery + md5 + lazy/work + overlay (self-contained)
src/background.js      page_action toggle + icon sync + cross-origin image fetch + defaults
src/popup.js           handshake-driven settings controls
```

No build step: the files load as-is. `src/content.js` is intentionally a single
self-contained script (manifest content scripts can't use ES imports). The
`icon-off*` PNGs are generated from our own `icon*` (desaturated + dimmed) — no
GPLv3 assets from ocr_extension are copied, keeping the tree clean-room.
