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

1. Start the server (`cd server && make serve`, default `http://127.0.0.1:4000`).
2. Click the Scanlation toolbar icon → confirm the **Server** URL → *Connect*
   (the dropdowns fill from the server handshake).
3. Pick languages / engines, then **Enable on tab**. Boxes appear over images;
   click a box to copy the original text; toggle *Show translated* for original.

## Files

```
manifest.json          MV3 manifest (action, service_worker, content_scripts)
content.css            overlay box styles
popup.html/css         popup UI
src/content.js         image discovery + md5 + lazy/work + overlay (self-contained)
src/service-worker.js  seeds default config into storage.local
src/popup.js           handshake-driven controls
icons/                 16/48/128
```

No build step: the files load as-is. `src/content.js` is intentionally a single
self-contained script (manifest content scripts can't use ES imports).
