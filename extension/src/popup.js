/* Scanlation popup (module). Handshake-driven controls; talks to the server
 * over fetch and to the active tab's content script via messaging. */
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

async function sendActive(msg) {
  try {
    const [tab] = await ext.tabs.query({ active: true, currentWindow: true });
    if (tab) await ext.tabs.sendMessage(tab.id, msg);
  } catch (e) { /* no content script on this tab */ }
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

    // labels = display names (detectors_hr…), value stays the engine id
    fillSelect($("detector"), d.detectors || [], d.detectors_hr, d.detector_selected);
    fillSelect($("recognizer"), d.recognizers || [], d.recognizers_hr, d.recognizer_selected);
    fillSelect($("translator"), d.translators || [], d.translators_hr, d.translator_selected);

    setStatus(`connected · v${(d.version || []).join(".")}`, "ok");

    await ext.storage.local.set({ endpoint: endpoint() });
    sendActive({ type: "set-endpoint", endpoint: endpoint() });
  } catch (e) {
    setStatus("cannot reach server: " + (e.message || e), "err");
  }
}

function wire() {
  $("connect").addEventListener("click", connect);

  $("lang_src").addEventListener("change", () =>
    post("/set_lang/", { lang_src: $("lang_src").value, lang_dst: $("lang_dst").value }));
  $("lang_dst").addEventListener("change", () =>
    post("/set_lang/", { lang_src: $("lang_src").value, lang_dst: $("lang_dst").value }));

  const setModels = () =>
    post("/set_models/", {
      detector: $("detector").value,
      recognizer: $("recognizer").value,
      translator: $("translator").value,
    });
  ["detector", "recognizer", "translator"].forEach((id) => $(id).addEventListener("change", setModels));

  $("enable").addEventListener("click", () => sendActive({ type: "enable" }));
  $("disable").addEventListener("click", () => sendActive({ type: "disable" }));

  $("showOriginal").addEventListener("change", async (e) => {
    const showTranslated = !e.target.checked;   // checkbox is "Show original"
    await ext.storage.local.set({ showTranslated });
    sendActive({ type: "set-show-translated", value: showTranslated });
  });
}

async function init() {
  const s = await ext.storage.local.get(["endpoint", "showTranslated"]);
  $("endpoint").value = s.endpoint || "http://127.0.0.1:4000";
  $("showOriginal").checked = s.showTranslated === false;   // checked = show original
  wire();
  connect(); // auto-connect to the saved endpoint
}

init();
