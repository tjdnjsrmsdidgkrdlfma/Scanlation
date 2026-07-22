"use strict";
// Shared config defaults for all three extension contexts (background event
// page, content isolated world, popup). Each context loads this classic script
// itself — content scripts and the event page can't ES-import, so it publishes
// onto globalThis for every realm to read. User-facing text lives in i18n.js.
globalThis.SCAN = {
  ENDPOINT: "http://127.0.0.1:4010",
  SHOW_TRANSLATED: true,
  MIN_IMAGE_DIM: 80,
};
