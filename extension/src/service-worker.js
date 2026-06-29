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
