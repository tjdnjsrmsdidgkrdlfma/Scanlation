"use strict";
// Shared defaults + user-facing strings for all three extension contexts
// (background event page, content isolated world, popup). Each context loads
// this classic script itself — content scripts and the event page can't
// ES-import, so it publishes onto globalThis for every realm to read.
globalThis.SCAN = {
  ENDPOINT: "http://127.0.0.1:4010",
  SHOW_TRANSLATED: true,
  MIN_IMAGE_DIM: 80,
  MSG_FAIL: "번역 실패",
};
