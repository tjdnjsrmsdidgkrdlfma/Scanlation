# Scanlation extension (MV3)

Clean-room Manifest V3 browser extension — no bundler, no npm, plain ES.
Overlays the Scanlation server's translations in place over manga images.

Wire contract matches the server exactly (verified end-to-end): md5 over the
**base64 string**, box `[x_min, y_min, x_max, y_max]`, `/run_ocrtsl/` lazy(md5)
→ work(contents) on a cache miss. The md5 implementation is byte-equal to
Python's `hashlib.md5`.

## Load unpacked

**Chrome / Edge** — `chrome://extensions` → enable *Developer mode* → *Load
unpacked* → select this `extension/` folder.

**Firefox** — `about:debugging#/runtime/this-firefox` → *Load Temporary Add-on*
→ pick `extension/manifest.json`.

## Use

1. Start the server (default `http://127.0.0.1:4010`).
2. **Settings** — right-click the toolbar icon → *Settings* (opens the options
   page). Confirm the **Server** URL → *Connect* (dropdowns fill from the
   handshake), pick languages / engines. (Model/lang/prompt can also be set
   server-side at `/admin`.)
3. **Translate** — **click the toolbar icon** to toggle translation on the
   active tab (one click, no popup). The icon **badge** shows state: `on` while
   active, then the count of translated regions. Click again to turn off.
   Click a box to copy the original text; *Show translated* (settings) flips
   between translated / original.

## Files

```
manifest.json          MV3 manifest (action=toggle, options_ui, service_worker, content_scripts)
content.css            overlay box styles
options.html/css       settings page (dark) — server, langs, engines, show-translated
src/content.js         image discovery + md5 + lazy/work + overlay (self-contained)
src/service-worker.js  icon-click toggle, badge, right-click Settings menu, image fetch, defaults
src/options.js         handshake-driven settings controls
icons/                 16/48/128
```

No build step: the files load as-is. `src/content.js` is intentionally a single
self-contained script (manifest content scripts can't use ES imports).
