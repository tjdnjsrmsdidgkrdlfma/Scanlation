/* Scanlation MV3 service worker (module).
 * Non-persistent: the only durable state is storage.local (single source of
 * truth for endpoint/config). The toolbar icon is a one-click toggle (no popup):
 * clicking it enables/disables translation on the active tab; the badge reflects
 * state + count. Settings live on the options page (right-click icon → Settings). */
const ext = globalThis.browser || globalThis.chrome;

const DEFAULTS = {
  endpoint: "http://127.0.0.1:4010",
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

  // Right-click the toolbar icon → Settings (opens the options page).
  try {
    ext.contextMenus.removeAll(() => {
      ext.contextMenus.create({ id: "settings", title: "Settings", contexts: ["action"] });
    });
  } catch (e) { /* contextMenus may be unavailable */ }
});

ext.contextMenus?.onClicked.addListener((info) => {
  if (info.menuItemId === "settings") ext.runtime.openOptionsPage();
});

// --- toolbar icon = one-click translate toggle ----------------------------
// The content script is auto-injected (content_scripts, <all_urls>). If it isn't
// there yet (e.g. a tab open from before install), inject it, then toggle.
async function toggleTab(tab) {
  if (!tab || !tab.id) return;
  try {
    await ext.tabs.sendMessage(tab.id, { type: "toggle" });
  } catch (e) {
    try {
      await ext.scripting.executeScript({ target: { tabId: tab.id }, files: ["src/content.js"] });
      await ext.scripting.insertCSS({ target: { tabId: tab.id }, files: ["content.css"] });
      await ext.tabs.sendMessage(tab.id, { type: "toggle" });
    } catch (e2) { /* restricted page (chrome://, addons store, etc.) */ }
  }
}
ext.action.onClicked.addListener(toggleTab);

// --- badge (state + count) from the content script ------------------------
function setBadge(tabId, enabled, count) {
  if (tabId == null) return;
  ext.action.setBadgeText({ tabId, text: enabled ? (count > 0 ? String(count) : "on") : "" });
  if (ext.action.setBadgeBackgroundColor) {
    ext.action.setBadgeBackgroundColor({ tabId, color: count > 0 ? "#4f7fd6" : "#8b93a3" });
  }
}

// --- cross-origin image fetch on behalf of the content script -------------
// The worker runs in the extension context with host_permissions, so it bypasses
// the page's CORS (it cannot defeat server-side Referer hotlink protection).
function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.slice(reader.result.indexOf(",") + 1));
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

ext.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.type === "badge") {
    setBadge(sender.tab && sender.tab.id, !!msg.enabled, msg.count || 0);
    return undefined;
  }
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
