/* Scanlation popup (module). Handshake-driven controls; talks to the server
 * over fetch and to the active tab's content script via messaging. */
const ext = globalThis.browser || globalThis.chrome;

const $ = (id) => document.getElementById(id);
const endpoint = () => SCANUTIL.trimEndpoint($("endpoint").value.trim());
const token = () => $("token").value.trim();
// Headers for a server call: JSON + the optional auth token (blank => omitted).
const authHeaders = (base) => SCANUTIL.authHeaders(base, token());

function setStatus(text, kind) {
  const el = $("status");
  el.textContent = text;
  el.className = "status" + (kind ? " " + kind : "");
}

// Fill every [data-i18n]/[data-i18n-ph] node and mark the active lang button.
function applyLang() {
  document.documentElement.lang = SCANI18N.lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = SCANI18N.t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-ph]").forEach((el) => { el.placeholder = SCANI18N.t(el.dataset.i18nPh); });
  document.querySelectorAll("#lang-toggle .langopt").forEach((el) => el.classList.toggle("active", el.dataset.lang === SCANI18N.lang));
}

// Persist the choice to ext.storage.local so the content script's failure badge
// shares it, and tell the active tab live. (A shown status line refreshes on the
// next Connect.)
async function selectLang(l) {
  SCANI18N.setLang(l);
  applyLang();
  try { await ext.storage.local.set({ lang: SCANI18N.lang }); } catch (e) { /* ignore */ }
  sendActive({ type: "set-lang", lang: SCANI18N.lang });
}

function fillSelect(sel, values, labels, selected) {
  sel.innerHTML = "";
  values.forEach((v, i) => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = labels ? labels[i] : v;
    if (v === selected) opt.selected = true;
    sel.appendChild(opt);
  });
}

async function sendActive(msg) {
  try {
    const [tab] = await ext.tabs.query({ active: true, currentWindow: true });
    if (tab) await ext.tabs.sendMessage(tab.id, msg);
  } catch (e) { /* no content script on this tab */ }
}

async function post(path, body) {
  const r = await fetch(endpoint() + path, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  return r.ok;
}

async function connect() {
  setStatus(SCANI18N.t("status.connecting"));
  try {
    const r = await fetch(endpoint() + "/", { headers: authHeaders() });
    if (r.status === 401) throw new Error(SCANI18N.t("status.authFail"));
    if (!r.ok) throw new Error("HTTP " + r.status);
    const d = await r.json();

    // languages: iso1 codes with human-readable labels (parallel arrays)
    const codeToName = {};
    (d.Languages || []).forEach((c, i) => (codeToName[c] = (d.Languages_hr || [])[i] || c));
    const nameOf = (codes) => codes.map((c) => codeToName[c] || c);
    fillSelect($("lang_src"), d.Languages_src || [], nameOf(d.Languages_src || []), d.lang_src);
    fillSelect($("lang_dst"), d.Languages_dst || [], nameOf(d.Languages_dst || []), d.lang_dst);

    // labels = display names (detectors_hr…), value stays the engine id
    fillSelect($("detector"), d.detectors || [], d.detectors_hr, d.detector_selected);
    fillSelect($("recognizer"), d.recognizers || [], d.recognizers_hr, d.recognizer_selected);
    fillSelect($("translator"), d.translators || [], d.translators_hr, d.translator_selected);

    setStatus(SCANI18N.t("status.connected", { v: (d.version || []).join(".") }), "ok");

    const store = { endpoint: endpoint(), token: token() };
    if (typeof d.min_image_dim === "number") store.minImageDim = d.min_image_dim;
    await ext.storage.local.set(store);
    sendActive({ type: "set-endpoint", endpoint: endpoint() });
    sendActive({ type: "set-token", token: token() });
    if (typeof d.min_image_dim === "number") sendActive({ type: "set-min-image-dim", value: d.min_image_dim });
  } catch (e) {
    setStatus(SCANI18N.t("status.unreachable", { msg: e.message || e }), "err");
  }
}

function wire() {
  $("connect").addEventListener("click", connect);

  $("lang_src").addEventListener("change", () =>
    post("/set_languages/", { lang_src: $("lang_src").value, lang_dst: $("lang_dst").value }));
  $("lang_dst").addEventListener("change", () =>
    post("/set_languages/", { lang_src: $("lang_src").value, lang_dst: $("lang_dst").value }));

  const setModels = () =>
    post("/set_engines/", {
      detector: $("detector").value,
      recognizer: $("recognizer").value,
      translator: $("translator").value,
    });
  ["detector", "recognizer", "translator"].forEach((id) => $(id).addEventListener("change", setModels));

  // Translate routes through the background so it injects the content script into
  // the active tab first (on-demand injection); Clear just messages an already-live tab.
  $("enable").addEventListener("click", async () => {
    const [tab] = await ext.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.id != null) Promise.resolve(ext.runtime.sendMessage({ type: "activate", tabId: tab.id })).catch(() => {});
  });
  $("disable").addEventListener("click", () => sendActive({ type: "disable" }));

  $("showOriginal").addEventListener("change", async (e) => {
    const showTranslated = !e.target.checked;   // checkbox is "Show original"
    await ext.storage.local.set({ showTranslated });
    sendActive({ type: "set-show-translated", value: showTranslated });
  });

  document.querySelectorAll("#lang-toggle .langopt")
    .forEach((el) => el.addEventListener("click", () => selectLang(el.dataset.lang)));
}

async function init() {
  const s = await ext.storage.local.get(["endpoint", "showTranslated", "token", "lang"]);
  $("endpoint").value = s.endpoint || globalThis.SCAN.ENDPOINT;
  $("token").value = s.token || "";
  $("showOriginal").checked = s.showTranslated === false;   // checked = show original
  SCANI18N.setLang(s.lang);
  applyLang();
  wire();
  connect(); // auto-connect to the saved endpoint
}

init();
