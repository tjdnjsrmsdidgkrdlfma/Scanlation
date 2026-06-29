/* Scanlation MV3 service worker (module).
 * Non-persistent: the only durable state is storage.local (single source of
 * truth for endpoint/config). The popup talks to the active tab's content
 * script directly, so the worker just seeds defaults on install. */
const ext = globalThis.browser || globalThis.chrome;

const DEFAULTS = {
  endpoint: "http://127.0.0.1:4000",
  showTranslated: true,
  fontScale: 1,
};

ext.runtime.onInstalled.addListener(async () => {
  const cur = await ext.storage.local.get(Object.keys(DEFAULTS));
  const patch = {};
  for (const [k, v] of Object.entries(DEFAULTS)) {
    if (cur[k] === undefined) patch[k] = v;
  }
  if (Object.keys(patch).length) await ext.storage.local.set(patch);
});

// Cross-origin image fetch on behalf of the content script. The worker runs in
// the extension context with host_permissions, so it bypasses the page's CORS
// (it cannot defeat server-side Referer hotlink protection, e.g. pixiv).
function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.slice(reader.result.indexOf(",") + 1));
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

ext.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.type === "fetch-image") {
    (async () => {
      try {
        const res = await fetch(msg.url);
        if (!res.ok) throw new Error("HTTP " + res.status);
        sendResponse({ ok: true, base64: await blobToBase64(await res.blob()) });
      } catch (e) {
        sendResponse({ ok: false, error: String(e && e.message ? e.message : e) });
      }
    })();
    return true; // keep the channel open for the async sendResponse
  }
  return undefined;
});
