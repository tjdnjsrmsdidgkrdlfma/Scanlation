"use strict";
// Scanlation admin — plain ES, no build step. The page is served at /admin/,
// but every API call targets the server root (leading slash), same origin.

const $ = (id) => document.getElementById(id);
let DATA = null; // last /get_settings/ snapshot

// --- tiny fetch helpers ---------------------------------------------------
async function api(path, opts) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch (_) { /* may be empty */ }
  if (!res.ok) throw new Error((body && body.detail) || `HTTP ${res.status}`);
  return body;
}
const postJSON = (path, data) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

function toast(msg, kind) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast " + (kind || "");
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, 2600);
}

// --- load + render --------------------------------------------------------
async function load() {
  try {
    DATA = await api("/get_settings/");
  } catch (e) {
    toast("서버 응답 없음: " + e.message, "err");
    return;
  }
  const v = DATA.version;
  if (Array.isArray(v)) $("version").textContent = "v" + v.join(".") + " · 서버 설정";
  $("app").hidden = false;
  $("tabbar").hidden = false;
  renderModels();
  renderLangs();
  renderPrompt();
  renderEngineOptions();
  renderPlugins();
}

function engineOption(e) {
  const mark = e.installed ? "●" : "○";
  return `<option value="${e.name}">${mark} ${e.display_name} (${e.name})</option>`;
}

function renderModels() {
  const roles = { detector: "sel-detector", recognizer: "sel-recognizer", translator: "sel-translator" };
  for (const [role, id] of Object.entries(roles)) {
    const sel = $(id);
    sel.innerHTML = DATA.engines[role].map(engineOption).join("");
    sel.value = DATA.selection[role];
  }
}

function renderLangs() {
  const langs = DATA.languages;
  const opts = Object.entries(langs).map(([k, hr]) => `<option value="${k}">${hr} (${k})</option>`).join("");
  $("sel-lang_src").innerHTML = opts;
  $("sel-lang_dst").innerHTML = opts;
  $("sel-lang_src").value = DATA.selection.lang_src;
  $("sel-lang_dst").value = DATA.selection.lang_dst;
}

// --- prompts --------------------------------------------------------------
function promptNames() {
  const builtin = Object.keys(DATA.prompts.builtin);
  const custom = Object.keys(DATA.prompts.custom).filter((n) => !DATA.prompts.builtin[n]);
  return { builtin, custom };
}
function promptText(name) {
  const c = DATA.prompts.custom, b = DATA.prompts.builtin;
  return (name in c) ? c[name] : (b[name] || "");
}
function isCustom(name) { return name in DATA.prompts.custom; }

function renderPrompt() {
  const { builtin, custom } = promptNames();
  const opt = (n, tag) => `<option value="${n}">${n}${tag}</option>`;
  let html = builtin.map((n) => opt(n, n in DATA.prompts.custom ? " (빌트인·수정됨)" : " (빌트인)")).join("");
  if (custom.length) html += custom.map((n) => opt(n, " (커스텀)")).join("");
  $("prompt-preset").innerHTML = html;
  $("prompt-preset").value = DATA.prompts.active;
  syncPromptEditor();
}
function syncPromptEditor() {
  const name = $("prompt-preset").value;
  $("prompt-text").value = promptText(name);
  $("prompt-name").value = name;
  $("prompt-delete").disabled = !isCustom(name);
}

// --- engine options -------------------------------------------------------
function findEngine(role, name) {
  return DATA.engines[role].find((e) => e.name === name);
}
function fieldInput(opt, spec, value) {
  const t = spec.type;
  const has = value !== undefined && value !== null;
  if (t === "bool") {
    const checked = (has ? value : spec.default) ? "checked" : "";
    return `<label class="opt-check"><input type="checkbox" data-opt="${opt}" data-type="bool" ${checked}/> ${opt}</label>`;
  }
  const inputType = (t === "int" || t === "float") ? "number" : "text";
  const step = t === "float" ? ` step="any"` : "";
  const val = has ? String(value) : "";
  const ph = spec.default === "" || spec.default === undefined ? "(기본값)" : `기본: ${spec.default}`;
  return `<label>${opt} <span class="desc">${spec.description || ""}</span>
    <input type="${inputType}"${step} data-opt="${opt}" data-type="${t}" value="${val}" placeholder="${ph}"/></label>`;
}
function optBlock(role, e) {
  const schema = e.schema || {};
  const keys = Object.keys(schema);
  let fields = `<p class="opt-empty">설정 가능한 옵션이 없습니다.</p>`;
  if (keys.length) {
    fields = `<div class="opt-fields">` +
      keys.map((k) => fieldInput(k, schema[k], e.options[k])).join("") +
      `</div><button class="btn primary sm" data-save-engine="${e.name}">저장</button>`;
  }
  return `<div class="opt-block" data-engine="${e.name}">
      <span class="role">${role}</span>
      <h3>${e.display_name} <span class="hint">${e.name}</span></h3>
      ${fields}
    </div>`;
}
function renderEngineOptions() {
  const sel = DATA.selection;
  const blocks = [
    optBlock("detector", findEngine("detector", sel.detector)),
    optBlock("recognizer", findEngine("recognizer", sel.recognizer)),
    optBlock("translator", findEngine("translator", sel.translator)),
  ];
  $("engine-options").innerHTML = blocks.join("");
}
function collectOptions(blockEl) {
  const out = {};
  blockEl.querySelectorAll("[data-opt]").forEach((el) => {
    const opt = el.dataset.opt, type = el.dataset.type;
    if (type === "bool") { out[opt] = el.checked; return; }
    const raw = el.value.trim();
    if (raw === "") { out[opt] = ""; return; } // "" -> server removes override (back to default)
    out[opt] = (type === "int") ? parseInt(raw, 10) : (type === "float") ? parseFloat(raw) : raw;
  });
  return out;
}

// --- plugins --------------------------------------------------------------
function renderPlugins() {
  const byName = {};
  for (const role of ["detector", "recognizer", "translator"]) {
    for (const e of DATA.engines[role]) {
      if (!byName[e.name]) byName[e.name] = { ...e, roles: [] };
      byName[e.name].roles.push(role);
    }
  }
  const rows = Object.values(byName).map((e) => {
    const badge = e.installed
      ? `<span class="pill pill-ok">설치됨</span>`
      : `<button class="btn sm primary" data-install="${e.name}">설치</button>`;
    const warn = e.warning ? `<div class="pwarn">⚠ ${e.warning}</div>` : "";
    return `<div class="plugin">
        <div class="meta">
          <div class="pname">${e.display_name} <span class="proles">${e.roles.join(", ")}</span></div>
          <div class="pdesc">${e.description || ""}</div>
          ${warn}
        </div>
        ${badge}
      </div>`;
  });
  $("plugins").innerHTML = rows.join("");
}

// --- actions --------------------------------------------------------------
async function saveModels() {
  try {
    await postJSON("/set_models/", {
      box_model_id: $("sel-detector").value,
      ocr_model_id: $("sel-recognizer").value,
      tsl_model_id: $("sel-translator").value,
    });
    await postJSON("/set_lang/", { lang_src: $("sel-lang_src").value, lang_dst: $("sel-lang_dst").value });
    toast("모델/언어 저장됨 — 이제 기본값입니다", "ok");
    await load();
  } catch (e) { toast("저장 실패: " + e.message, "err"); }
}

async function usePrompt() {
  try {
    await postJSON("/select_prompt/", { name: $("prompt-preset").value });
    toast("프롬프트 적용됨", "ok");
    await load();
  } catch (e) { toast("실패: " + e.message, "err"); }
}
async function savePrompt() {
  const name = $("prompt-name").value.trim();
  if (!name) { toast("저장 이름을 입력하세요", "err"); return; }
  try {
    await postJSON("/save_prompt/", { name, text: $("prompt-text").value });
    toast(`"${name}" 저장 & 적용됨`, "ok");
    await load();
  } catch (e) { toast("실패: " + e.message, "err"); }
}
async function deletePrompt() {
  const name = $("prompt-preset").value;
  if (!confirm(`커스텀 프롬프트 "${name}" 삭제?`)) return;
  try {
    await postJSON("/delete_prompt/", { name });
    toast(`"${name}" 삭제됨`, "ok");
    await load();
  } catch (e) { toast("실패: " + e.message, "err"); }
}

async function saveEngineOptions(engine, blockEl) {
  try {
    await postJSON("/set_options/", { engine, options: collectOptions(blockEl) });
    toast(`${engine} 옵션 저장됨`, "ok");
    await load();
  } catch (e) { toast("실패: " + e.message, "err"); }
}
async function installPlugin(name) {
  toast(`${name} 설치 중… (가중치 다운로드)`, "");
  try {
    const r = await postJSON("/manage_plugins/", { plugins: { [name]: true } });
    toast(`${name} ${r.status === "success" ? "설치 완료" : "결과: " + r.status}`, "ok");
    await load();
  } catch (e) { toast(`${name} 설치 실패: ` + e.message, "err"); }
}

// --- tabs -----------------------------------------------------------------
function showView(name) {
  document.querySelectorAll(".view").forEach((el) => el.classList.toggle("active", el.dataset.view === name));
  document.querySelectorAll(".tab").forEach((el) => el.classList.toggle("active", el.dataset.view === name));
}

// --- wire up --------------------------------------------------------------
$("tabbar").addEventListener("click", (ev) => {
  const t = ev.target.closest(".tab");
  if (t) showView(t.dataset.view);
});
$("save-models").addEventListener("click", saveModels);
$("prompt-preset").addEventListener("change", syncPromptEditor);
$("prompt-use").addEventListener("click", usePrompt);
$("prompt-save").addEventListener("click", savePrompt);
$("prompt-delete").addEventListener("click", deletePrompt);
$("engine-options").addEventListener("click", (ev) => {
  const btn = ev.target.closest("[data-save-engine]");
  if (!btn) return;
  saveEngineOptions(btn.dataset.saveEngine, btn.closest(".opt-block"));
});
$("plugins").addEventListener("click", (ev) => {
  const btn = ev.target.closest("[data-install]");
  if (btn) installPlugin(btn.dataset.install);
});

load();
