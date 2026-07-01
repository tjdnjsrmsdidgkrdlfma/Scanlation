/* Scanlation options page (module). Settings only — connects to the server,
 * persists endpoint/langs/engines/showTranslated. Translation on/off is the
 * toolbar icon's job (one-click toggle), not this page. Since this runs in its
 * own tab, live changes are broadcast to every tab's content script. */
const ext = globalThis.browser || globalThis.chrome;

const $ = (id) => document.getElementById(id);
const endpoint = () => $("endpoint").value.trim().replace(/\/$/, "");

function setStatus(text, kind) {
  const el = $("status");
  el.textContent = text;
  el.className = "status" + (kind ? " " + kind : "");
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

// Broadcast a message to every tab's content script (this page is its own tab).
async function sendAll(msg) {
  try {
    const tabs = await ext.tabs.query({});
    for (const tab of tabs) {
      try { await ext.tabs.sendMessage(tab.id, msg); } catch (e) { /* no content script here */ }
    }
  } catch (e) { /* tabs unavailable */ }
}

async function post(path, body) {
  const r = await fetch(endpoint() + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.ok;
}

async function connect() {
  setStatus("connecting…");
  try {
    const r = await fetch(endpoint() + "/");
    if (!r.ok) throw new Error("HTTP " + r.status);
    const d = await r.json();

    // languages: iso1 codes with human-readable labels (parallel arrays)
    const codeToName = {};
    (d.Languages || []).forEach((c, i) => (codeToName[c] = (d.Languages_hr || [])[i] || c));
    const nameOf = (codes) => codes.map((c) => codeToName[c] || c);
    fillSelect($("lang_src"), d.Languages_src || [], nameOf(d.Languages_src || []), d.lang_src);
    fillSelect($("lang_dst"), d.Languages_dst || [], nameOf(d.Languages_dst || []), d.lang_dst);

    fillSelect($("detector"), d.detectors || [], null, d.detector_selected);
    fillSelect($("recognizer"), d.recognizers || [], null, d.recognizer_selected);
    fillSelect($("translator"), d.translators || [], null, d.translator_selected);

    $("conn").hidden = false;
    setStatus(`connected · v${(d.version || []).join(".")}`, "ok");

    await ext.storage.local.set({ endpoint: endpoint() });
    sendAll({ type: "set-endpoint", endpoint: endpoint() });
  } catch (e) {
    $("conn").hidden = true;
    setStatus("cannot reach server: " + (e.message || e), "err");
  }
}

function wire() {
  $("connect").addEventListener("click", connect);

  const setLang = () =>
    post("/set_lang/", { lang_src: $("lang_src").value, lang_dst: $("lang_dst").value });
  $("lang_src").addEventListener("change", setLang);
  $("lang_dst").addEventListener("change", setLang);

  const setModels = () =>
    post("/set_models/", {
      detector: $("detector").value,
      recognizer: $("recognizer").value,
      translator: $("translator").value,
    });
  ["detector", "recognizer", "translator"].forEach((id) => $(id).addEventListener("change", setModels));

  $("showTranslated").addEventListener("change", async (e) => {
    await ext.storage.local.set({ showTranslated: e.target.checked });
    sendAll({ type: "set-show-translated", value: e.target.checked });
  });
}

async function init() {
  const s = await ext.storage.local.get(["endpoint", "showTranslated"]);
  $("endpoint").value = s.endpoint || "http://127.0.0.1:4010";
  $("showTranslated").checked = s.showTranslated !== false;
  wire();
  connect(); // auto-connect to the saved endpoint
}

init();
