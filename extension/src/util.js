"use strict";
// Pure, DOM-light helpers shared across all three extension contexts (content
// isolated world, background event page, popup). Like constants.js this is a
// classic script — content scripts and the event page can't ES-import — so it
// publishes onto globalThis for every realm to read. No DOM or extension APIs
// are touched at load time, so it is also loadable standalone (see tests/run.mjs).
globalThis.SCANUTIL = {
  // Strip a trailing slash so a configured endpoint can't produce a `//path` URL.
  trimEndpoint(url) {
    return String(url == null ? "" : url).replace(/\/$/, "");
  },

  // JSON-call headers plus the optional auth token (blank token => header omitted).
  authHeaders(base, token) {
    const h = base ? { ...base } : {};
    if (token) h["X-Auth-Token"] = token;
    return h;
  },

  // A Blob's base64 payload with the "data:...;base64," prefix stripped.
  blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result.slice(reader.result.indexOf(",") + 1));
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  },

  // box [x_min, y_min, x_max, y_max] in natural px -> fractions (0-1) of the
  // natural size, so they map onto either % CSS or displayed-px equally.
  boxFractions(box, nw, nh) {
    const [l, b, r, t] = box;
    return { left: l / nw, top: b / nh, w: (r - l) / nw, h: (t - b) / nh };
  },
};
