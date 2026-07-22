/* Scanlation MV2 background (event page — Firefox).
 *
 * Two toolbar surfaces, like the old ocr_extension:
 *   - browser_action  -> settings popup (popup.html)
 *   - page_action     -> address-bar icon; one click toggles translation on the tab
 *
 * The content script owns the enable state; the page_action icon FOLLOWS its
 * "state" reports, so the icon stays correct whether toggled from the address-bar
 * icon or the popup's Enable/Disable. Only durable state is storage.local. */
const ext = globalThis.browser || globalThis.chrome;

const DEFAULTS = {
  endpoint: globalThis.SCAN.ENDPOINT,
  showTranslated: globalThis.SCAN.SHOW_TRANSLATED,
  minImageDim: globalThis.SCAN.MIN_IMAGE_DIM,
};
const ICON_ON = { 16: "icons/icon16.png", 48: "icons/icon48.png" };
const ICON_OFF = { 16: "icons/icon-off16.png", 48: "icons/icon-off48.png" };
const TITLE_ON = "Scanlation: translating — click to stop";
const TITLE_OFF = "Scanlation: click to translate this tab";

ext.runtime.onInstalled.addListener(async () => {
  const cur = await ext.storage.local.get(Object.keys(DEFAULTS));
  const patch = {};
  for (const [k, v] of Object.entries(DEFAULTS)) if (cur[k] === undefined) patch[k] = v;
  if (Object.keys(patch).length) await ext.storage.local.set(patch);
});

// address-bar icon click = one-click translate toggle
ext.pageAction.onClicked.addListener((tab) => {
  if (tab && tab.id != null) {
    Promise.resolve(ext.tabs.sendMessage(tab.id, { type: "toggle" })).catch(() => {});
  }
});

function setPageAction(tabId, enabled) {
  if (tabId == null) return;
  try {
    ext.pageAction.setIcon({ tabId, path: enabled ? ICON_ON : ICON_OFF });
    ext.pageAction.setTitle({ tabId, title: enabled ? TITLE_ON : TITLE_OFF });
  } catch (e) { /* tab gone */ }
}

// Cross-origin image fetch on behalf of the content script (extension context
// has host permissions, so it bypasses page CORS; cannot beat Referer hotlinking).
ext.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.type === "state") {
    setPageAction(sender.tab && sender.tab.id, !!msg.enabled);
    return undefined;
  }
  if (msg && msg.type === "fetch-image") {
    (async () => {
      try {
        const res = await fetch(msg.url);
        if (!res.ok) throw new Error("HTTP " + res.status);
        sendResponse({ ok: true, base64: await SCANUTIL.blobToBase64(await res.blob()) });
      } catch (e) {
        sendResponse({ ok: false, error: String(e && e.message ? e.message : e) });
      }
    })();
    return true; // keep the channel open for the async sendResponse
  }
  return undefined;
});
