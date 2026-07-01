"use strict";
// Scanlation admin — plain ES, no build step. The page is served at /admin/,
// but every API call targets the server root (leading slash), same origin.

const $ = (id) => document.getElementById(id);
let DATA = null; // last /get_settings/ snapshot

// --- i18n -----------------------------------------------------------------
// UI chrome only. Engine metadata (display_name/description/warning/option
// descriptions) comes from the server already in English, so it isn't keyed
// here. `{x}` tokens are filled by t(key, {x: ...}).
const I18N = {
  ko: {
    "brand.sub": "서버 설정 · 모델 · 프롬프트",
    "version.suffix": "서버 설정",
    "tab.models": "모델 · 언어",
    "tab.prompt": "번역 프롬프트",
    "tab.options": "엔진 옵션",
    "tab.plugins": "플러그인",
    "models.h2": "모델 선택",
    "models.hint": "마지막 선택이 서버에 저장되어 기본값이 됩니다",
    "models.detector": "박스(detector)",
    "models.recognizer": "OCR(recognizer)",
    "models.translator": "번역(translator)",
    "models.src": "원문",
    "models.dst": "번역",
    "btn.save": "저장",
    "prompt.h2": "번역 프롬프트",
    "prompt.hint": "LLM 시스템 프롬프트를 고르거나 직접 편집",
    "prompt.preset": "프리셋",
    "prompt.use": "이 프롬프트 사용",
    "prompt.delete": "삭제",
    "prompt.saveName": "저장 이름",
    "prompt.namePlaceholder": "예: my-natural",
    "prompt.save": "저장 & 사용",
    "prompt.note": "빌트인을 편집해 같은 이름으로 저장하면 그 이름의 커스텀 프리셋이 덮어씁니다. 새 이름으로 저장하면 복제됩니다. 커스텀을 삭제하면 빌트인으로 되돌아갑니다.",
    "options.h2": "엔진 옵션",
    "options.hint": "선택된 엔진의 옵션 (예: 번역기 모델 태그)",
    "options.none": "설정 가능한 옵션이 없습니다.",
    "plugins.h2": "플러그인 설치",
    "plugins.hint": "가중치/모델 원클릭 설치",
    "plugins.installed": "설치됨",
    "plugins.install": "설치",
    "field.default": "(기본값)",
    "field.defaultPrefix": "기본",
    "preset.builtin": "빌트인",
    "preset.builtinEdited": "빌트인·수정됨",
    "preset.custom": "커스텀",
    "toast.noServer": "서버 응답 없음: {msg}",
    "toast.modelsSaved": "모델/언어 저장됨 — 이제 기본값입니다",
    "toast.saveFail": "저장 실패: {msg}",
    "toast.promptApplied": "프롬프트 적용됨",
    "toast.fail": "실패: {msg}",
    "toast.enterName": "저장 이름을 입력하세요",
    "toast.savedApplied": "\"{name}\" 저장 & 적용됨",
    "confirm.deletePrompt": "커스텀 프롬프트 \"{name}\" 삭제?",
    "toast.deleted": "\"{name}\" 삭제됨",
    "toast.optionsSaved": "{engine} 옵션 저장됨",
    "toast.installing": "{name} 설치 중… (가중치 다운로드)",
    "toast.installed": "{name} 설치 완료",
    "toast.installResult": "{name} 결과: {status}",
    "toast.installFail": "{name} 설치 실패: {msg}",
  },
  en: {
    "brand.sub": "Server settings · models · prompts",
    "version.suffix": "server settings",
    "tab.models": "Models · Language",
    "tab.prompt": "Prompt",
    "tab.options": "Engine options",
    "tab.plugins": "Plugins",
    "models.h2": "Model selection",
    "models.hint": "Your last choice is saved on the server as the default",
    "models.detector": "Boxes (detector)",
    "models.recognizer": "OCR (recognizer)",
    "models.translator": "Translation (translator)",
    "models.src": "Source",
    "models.dst": "Target",
    "btn.save": "Save",
    "prompt.h2": "Translation prompt",
    "prompt.hint": "Pick or edit the LLM system prompt",
    "prompt.preset": "Preset",
    "prompt.use": "Use this prompt",
    "prompt.delete": "Delete",
    "prompt.saveName": "Save as",
    "prompt.namePlaceholder": "e.g. my-natural",
    "prompt.save": "Save & use",
    "prompt.note": "Editing a builtin and saving under the same name creates a custom preset that overrides it. Saving under a new name clones it. Deleting a custom preset reverts to the builtin.",
    "options.h2": "Engine options",
    "options.hint": "Options for the selected engines (e.g. translator model tag)",
    "options.none": "No configurable options.",
    "plugins.h2": "Plugin installation",
    "plugins.hint": "One-click install of weights/models",
    "plugins.installed": "Installed",
    "plugins.install": "Install",
    "field.default": "(default)",
    "field.defaultPrefix": "default",
    "preset.builtin": "builtin",
    "preset.builtinEdited": "builtin · edited",
    "preset.custom": "custom",
    "toast.noServer": "No server response: {msg}",
    "toast.modelsSaved": "Models/language saved — now the default",
    "toast.saveFail": "Save failed: {msg}",
    "toast.promptApplied": "Prompt applied",
    "toast.fail": "Failed: {msg}",
    "toast.enterName": "Enter a name to save",
    "toast.savedApplied": "\"{name}\" saved & applied",
    "confirm.deletePrompt": "Delete custom prompt \"{name}\"?",
    "toast.deleted": "\"{name}\" deleted",
    "toast.optionsSaved": "{engine} options saved",
    "toast.installing": "Installing {name}… (downloading weights)",
    "toast.installed": "{name} installed",
    "toast.installResult": "{name} result: {status}",
    "toast.installFail": "{name} install failed: {msg}",
  },
};

const LANG_KEY = "scan_admin_lang";
function initLang() {
  let saved = null;
  try { saved = localStorage.getItem(LANG_KEY); } catch (_) { /* private mode */ }
  if (saved === "ko" || saved === "en") return saved;
  return (navigator.language || "").toLowerCase().startsWith("ko") ? "ko" : "en";
}
let LANG = initLang();

function t(key, vars) {
  let s = (I18N[LANG] && I18N[LANG][key]) || I18N.ko[key] || key;
  if (vars) for (const k in vars) s = s.split("{" + k + "}").join(vars[k]);
  return s;
}
function setLang(l) {
  LANG = l;
  try { localStorage.setItem(LANG_KEY, l); } catch (_) { /* ignore */ }
}
function applyLang() {
  document.documentElement.lang = LANG;
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-ph]").forEach((el) => { el.placeholder = t(el.dataset.i18nPh); });
  $("lang-toggle").textContent = LANG === "ko" ? "EN" : "KO";
}

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
  const el = $("toast");
  el.textContent = msg;
  el.className = "toast " + (kind || "");
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, 2600);
}

// --- load + render --------------------------------------------------------
async function load() {
  try {
    DATA = await api("/get_settings/");
  } catch (e) {
    toast(t("toast.noServer", { msg: e.message }), "err");
    return;
  }
  render();
}
// Re-render everything from the cached DATA (no refetch). Called on load and
// whenever the language toggles, so dynamically-built strings re-localize.
function render() {
  const v = DATA.version;
  if (Array.isArray(v)) $("version").textContent = "v" + v.join(".") + " · " + t("version.suffix");
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
  const opt = (n, tag) => `<option value="${n}">${n} (${tag})</option>`;
  let html = builtin.map((n) => opt(n, n in DATA.prompts.custom ? t("preset.builtinEdited") : t("preset.builtin"))).join("");
  if (custom.length) html += custom.map((n) => opt(n, t("preset.custom"))).join("");
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
  const type = spec.type;
  const has = value !== undefined && value !== null;
  if (type === "bool") {
    const checked = (has ? value : spec.default) ? "checked" : "";
    return `<label class="opt-check"><input type="checkbox" data-opt="${opt}" data-type="bool" ${checked}/> ${opt}</label>`;
  }
  const inputType = (type === "int" || type === "float") ? "number" : "text";
  const step = type === "float" ? ` step="any"` : "";
  const val = has ? String(value) : "";
  const ph = spec.default === "" || spec.default === undefined ? t("field.default") : `${t("field.defaultPrefix")}: ${spec.default}`;
  return `<label>${opt} <span class="desc">${spec.description || ""}</span>
    <input type="${inputType}"${step} data-opt="${opt}" data-type="${type}" value="${val}" placeholder="${ph}"/></label>`;
}
function optBlock(role, e) {
  const schema = e.schema || {};
  const keys = Object.keys(schema);
  let fields = `<p class="opt-empty">${t("options.none")}</p>`;
  if (keys.length) {
    fields = `<div class="opt-fields">` +
      keys.map((k) => fieldInput(k, schema[k], e.options[k])).join("") +
      `</div><button class="btn primary sm" data-save-engine="${e.name}">${t("btn.save")}</button>`;
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
      ? `<span class="pill pill-ok">${t("plugins.installed")}</span>`
      : `<button class="btn sm primary" data-install="${e.name}">${t("plugins.install")}</button>`;
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
    toast(t("toast.modelsSaved"), "ok");
    await load();
  } catch (e) { toast(t("toast.saveFail", { msg: e.message }), "err"); }
}

async function usePrompt() {
  try {
    await postJSON("/select_prompt/", { name: $("prompt-preset").value });
    toast(t("toast.promptApplied"), "ok");
    await load();
  } catch (e) { toast(t("toast.fail", { msg: e.message }), "err"); }
}
async function savePrompt() {
  const name = $("prompt-name").value.trim();
  if (!name) { toast(t("toast.enterName"), "err"); return; }
  try {
    await postJSON("/save_prompt/", { name, text: $("prompt-text").value });
    toast(t("toast.savedApplied", { name }), "ok");
    await load();
  } catch (e) { toast(t("toast.fail", { msg: e.message }), "err"); }
}
async function deletePrompt() {
  const name = $("prompt-preset").value;
  if (!confirm(t("confirm.deletePrompt", { name }))) return;
  try {
    await postJSON("/delete_prompt/", { name });
    toast(t("toast.deleted", { name }), "ok");
    await load();
  } catch (e) { toast(t("toast.fail", { msg: e.message }), "err"); }
}

async function saveEngineOptions(engine, blockEl) {
  try {
    await postJSON("/set_options/", { engine, options: collectOptions(blockEl) });
    toast(t("toast.optionsSaved", { engine }), "ok");
    await load();
  } catch (e) { toast(t("toast.fail", { msg: e.message }), "err"); }
}
async function installPlugin(name) {
  toast(t("toast.installing", { name }), "");
  try {
    const r = await postJSON("/manage_plugins/", { plugins: { [name]: true } });
    toast(r.status === "success" ? t("toast.installed", { name }) : t("toast.installResult", { name, status: r.status }), "ok");
    await load();
  } catch (e) { toast(t("toast.installFail", { name, msg: e.message }), "err"); }
}

// --- tabs -----------------------------------------------------------------
function showView(name) {
  document.querySelectorAll(".view").forEach((el) => el.classList.toggle("active", el.dataset.view === name));
  document.querySelectorAll(".tab").forEach((el) => el.classList.toggle("active", el.dataset.view === name));
}

// --- wire up --------------------------------------------------------------
$("lang-toggle").addEventListener("click", () => {
  setLang(LANG === "ko" ? "en" : "ko");
  applyLang();
  if (DATA) render();
});
$("tabbar").addEventListener("click", (ev) => {
  const tab = ev.target.closest(".tab");
  if (tab) showView(tab.dataset.view);
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

applyLang();
load();
