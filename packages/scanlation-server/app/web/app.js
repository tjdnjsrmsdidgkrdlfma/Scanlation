"use strict";
// Scanlation admin — plain ES, no build step. The page is served at /admin/,
// but every API call targets the server root (leading slash), same origin.

const $ = (id) => document.getElementById(id);
let DATA = null; // last /get_settings/ snapshot

// The three engine roles, in fixed order (mirrors the server's ROLE_NAMES).
const ROLES = ["detector", "recognizer", "translator"];
// An engine whose pip package is present (catalog-only entries send
// installed_package:false and aren't selectable until installed).
const pkgInstalled = (e) => e.installed_package !== false;

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
    "options.hint": "선택된 엔진의 옵션 (예: TSL 모델 태그)",
    "options.none": "설정 가능한 옵션이 없습니다.",
    // engine option descriptions (ko override of the server's English schema desc)
    "opt.temperature": "샘플링 온도 (0 = 결정적)",
    "opt.seed": "난수 시드",
    "opt.top_p": "누클리어스 샘플링 p",
    "opt.ctd.det_size": "정사각 추론 크기 (레터박스)",
    "opt.ctd.mask_threshold": "마스크 이진화 임계값",
    "opt.ctd.min_area": "이보다 작은 박스 제거 (원본 px²)",
    "opt.ctd.min_side": "짧은 변이 이보다 얇은 박스 제거 (원본 px) — SFX 조각·말줄임표 컷",
    "opt.ctd.unclip_ratio": "쿼드를 바깥으로 팽창 (1.0 = 안 함)",
    "opt.ctd.merge_px": "글자를 줄/말풍선으로 병합하는 모폴로지 커널 (마스크 px); 0 = 글자별",
    "opt.ctd.merge_aspect": "병합 커널 세로/가로 비 (>1이면 세로 컬럼 방향으로 병합, 컬럼끼리 유지)",
    "opt.ollama.model": "ollama 모델 태그 (예: gemma4:31b). 필수 — /admin에서 선택",
    "opt.ollama.num_ctx": "KV 캐시 컨텍스트 창 (번역 입력은 짧음)",
    "opt.ollama.num_gpu": "GPU로 오프로드할 레이어 수",
    "opt.llamacpp.model": "모델 id (서버 /v1/models). 필수 — /admin에서 선택",
    "opt.llamacpp.max_tokens": "생성 최대 토큰 수",
    "plugins.h2": "플러그인 설치",
    "plugins.hint": "패키지 + 가중치 원클릭 설치",
    "plugins.installed": "설치됨",
    "plugins.install": "설치",
    "plugins.installWeights": "가중치 설치",
    "plugins.notInstalledPkg": "패키지 미설치",
    "tab.maintenance": "유지보수",
    "maint.h2": "유지보수",
    "maint.hint": "캐시 관리",
    "maint.cache.title": "캐시 비우기",
    "maint.cache.desc": "저장된 모든 캐시(페이지 결과 + 번역 기록)를 지워 다음 접속 때 전 과정(검출·인식·번역)을 새로 실행합니다.",
    "maint.cache.btn": "캐시 비우기",
    "field.default": "(기본값)",
    "field.defaultPrefix": "기본",
    "field.pickModel": "— 모델 선택 —",
    "field.notInstalled": "(미설치)",
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
    "confirm.clearCache": "모든 캐시(페이지 결과·번역 기록)를 비울까요?",
    "toast.cacheCleared": "캐시 비움 — {n}건",
    "toast.deleted": "\"{name}\" 삭제됨",
    "toast.optionsSaved": "{engine} 옵션 저장됨",
    "toast.installing": "{name} 설치 중… (패키지 + 가중치, 오래 걸릴 수 있음)",
    "toast.installed": "{name} 설치 완료",
    "toast.installResult": "{name} 결과: {status}",
    "toast.installFail": "{name} 설치 실패: {msg}",
    "auth.tokenPh": "X-Auth-Token (없으면 비움)",
    "gate.title": "관리자 인증",
    "gate.hint": "이 서버는 접근 토큰이 필요합니다.",
    "gate.submit": "입력",
    "gate.wrong": "토큰이 올바르지 않습니다.",
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
    "options.hint": "Options for the selected engines (e.g. TSL model tag)",
    "options.none": "No configurable options.",
    "plugins.h2": "Plugin installation",
    "plugins.hint": "One-click install of package + weights",
    "plugins.installed": "Installed",
    "plugins.install": "Install",
    "plugins.installWeights": "Install weights",
    "plugins.notInstalledPkg": "package not installed",
    "tab.maintenance": "Maintenance",
    "maint.h2": "Maintenance",
    "maint.hint": "Cache management",
    "maint.cache.title": "Clear cache",
    "maint.cache.desc": "Removes all cached data (page results + translation log) so the full pipeline (detect · recognize · translate) re-runs next time.",
    "maint.cache.btn": "Clear cache",
    "field.default": "(default)",
    "field.defaultPrefix": "default",
    "field.pickModel": "— pick a model —",
    "field.notInstalled": "(not installed)",
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
    "confirm.clearCache": "Clear all cache (page results + translation log)?",
    "toast.cacheCleared": "Cache cleared — {n} entrie(s)",
    "toast.deleted": "\"{name}\" deleted",
    "toast.optionsSaved": "{engine} options saved",
    "toast.installing": "Installing {name}… (package + weights, may take a while)",
    "toast.installed": "{name} installed",
    "toast.installResult": "{name} result: {status}",
    "toast.installFail": "{name} install failed: {msg}",
    "auth.tokenPh": "X-Auth-Token (blank if none)",
    "gate.title": "Admin access",
    "gate.hint": "This server requires an access token.",
    "gate.submit": "Enter",
    "gate.wrong": "Invalid token.",
  },
};

const LANG_KEY = "scan_admin_lang";
function initLang() {
  let saved = null;
  try { saved = localStorage.getItem(LANG_KEY); } catch (_) { /* private mode */ }
  return (saved === "ko" || saved === "en") ? saved : "en"; // English is the default
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
  document.querySelectorAll("#lang-toggle .langopt").forEach((el) => el.classList.toggle("active", el.dataset.lang === LANG));
}

// --- auth token (optional; sent as X-Auth-Token) --------------------------
const TOKEN_KEY = "scan_admin_token";
const getToken = () => { try { return localStorage.getItem(TOKEN_KEY) || ""; } catch (_) { return ""; } };
const setToken = (v) => { try { localStorage.setItem(TOKEN_KEY, v); } catch (_) { /* ignore */ } };

// --- tiny fetch helpers ---------------------------------------------------
async function api(path, opts) {
  const o = { ...(opts || {}) };
  const tok = getToken();
  o.headers = { ...(o.headers || {}), ...(tok ? { "X-Auth-Token": tok } : {}) };
  const res = await fetch(path, o);
  let body = null;
  try { body = await res.json(); } catch (_) { /* may be empty */ }
  if (!res.ok) {
    const err = new Error((body && body.detail) || `HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
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
// The login gate covers the page until we hold a token the server accepts. It
// only ever appears when the server actually requires one (401); with auth off
// (or a valid stored token) load() succeeds straight away and the gate stays hidden.
function showGate(wrong) {
  $("gate").hidden = false;
  $("gate-err").hidden = !wrong;
  const el = $("gate-token");
  el.value = getToken();
  el.focus();
}
function hideGate() { $("gate").hidden = true; }

async function load({ wrong = false } = {}) {
  try {
    DATA = await api("/get_settings/");
  } catch (e) {
    if (e.status === 401) { showGate(wrong); return; }
    toast(t("toast.noServer", { msg: e.message }), "err");
    return;
  }
  hideGate();
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
  for (const role of ROLES) {
    const sel = $("sel-" + role);
    // Only pip-installed engines are selectable (catalog-only ones aren't in the
    // registry yet; install them in the plugin tab first).
    const installed = DATA.engines[role].filter(pkgInstalled);
    sel.innerHTML = installed.map(engineOption).join("");
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
// Localized option description: `opt.<engine>.<key>` then a generic `opt.<key>`
// override from I18N[LANG], else the server's (English) schema description. Only
// ko carries overrides, so en falls straight through to the server string.
function optDesc(engine, opt, fallback) {
  const m = I18N[LANG];
  const v = m && (m["opt." + engine + "." + opt] || m["opt." + opt]);
  return v || fallback || "";
}
function fieldInput(engine, opt, spec, value) {
  const type = spec.type;
  const has = value !== undefined && value !== null;
  if (type === "bool") {
    const checked = (has ? value : spec.default) ? "checked" : "";
    return `<label class="opt-check"><input type="checkbox" data-opt="${opt}" data-type="bool" ${checked}/><span class="toggle"></span> ${opt}</label>`;
  }
  const inputType = (type === "int" || type === "float") ? "number" : "text";
  const step = type === "float" ? ` step="any"` : "";
  const val = has ? String(value) : "";
  const ph = spec.default === "" || spec.default === undefined ? t("field.default") : `${t("field.defaultPrefix")}: ${spec.default}`;
  return `<label>${opt} <span class="desc">${optDesc(engine, opt, spec.description)}</span>
    <input type="${inputType}"${step} data-opt="${opt}" data-type="${type}" value="${val}" placeholder="${ph}"/></label>`;
}
function optBlock(role, e) {
  const schema = e.schema || {};
  const keys = Object.keys(schema);
  let fields = `<p class="opt-empty">${t("options.none")}</p>`;
  if (keys.length) {
    fields = `<div class="opt-fields">` +
      keys.map((k) => fieldInput(e.name, k, schema[k], e.options[k])).join("") +
      `</div><button class="btn primary sm" data-save-engine="${e.name}">${t("btn.save")}</button>`;
  }
  return `<div class="opt-block" data-engine="${e.name}">
      <span class="role">${role}</span>
      <h3>${e.display_name}</h3>
      ${fields}
    </div>`;
}
function renderEngineOptions() {
  const sel = DATA.selection;
  const blocks = ROLES.map((role) => optBlock(role, findEngine(role, sel[role])));
  $("engine-options").innerHTML = blocks.join("");
  setupModelPicker();  // async: swap the translator 'model' text field for a <select>
}

// Replace the active translator's free-text 'model' field with a <select> of the
// backend's installed models, plus a "(default — env)" option (value "") that
// clears the override. A saved value not in the list is kept as an option. If the
// backend is unreachable (empty list) the plain text input is left untouched, so
// a tag can still be typed.
async function setupModelPicker() {
  const engine = DATA.selection.translator;
  const block = document.querySelector(`.opt-block[data-engine="${engine}"]`);
  const input = block && block.querySelector('input[data-opt="model"]');
  if (!input) return;  // dummy translator has no model option
  let models = [];
  try {
    models = (await api(`/get_translator_models/?engine=${encodeURIComponent(engine)}`)).models || [];
  } catch (_) { return; }
  if (!models.length) return;  // backend down -> keep the text input
  const current = input.value;
  const opts = [`<option value="">${t("field.pickModel")}</option>`];
  if (current && !models.includes(current)) {
    opts.push(`<option value="${current}">${current} ${t("field.notInstalled")}</option>`);
  }
  for (const m of models) opts.push(`<option value="${m}">${m}</option>`);
  const sel = document.createElement("select");
  sel.dataset.opt = "model";
  sel.dataset.type = "str";
  sel.innerHTML = opts.join("");
  sel.value = current;
  input.replaceWith(sel);
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
  for (const role of ROLES) {
    for (const e of DATA.engines[role]) {
      if (!byName[e.name]) byName[e.name] = { ...e, roles: [] };
      byName[e.name].roles.push(role);
    }
  }
  const rows = Object.values(byName).map((e) => {
    // Two install layers: package (pip) then weights. Show the next needed step.
    let action;
    if (!pkgInstalled(e)) {
      action = `<button class="btn sm primary" data-install="${e.name}">${t("plugins.install")}</button>`;
    } else if (!e.installed) {
      action = `<button class="btn sm primary" data-install="${e.name}">${t("plugins.installWeights")}</button>`;
    } else {
      action = `<span class="pill pill-ok">${t("plugins.installed")}</span>`;
    }
    const pkgTag = !pkgInstalled(e) ? ` <span class="proles">${t("plugins.notInstalledPkg")}</span>` : "";
    const warn = e.warning ? `<div class="pwarn">⚠ ${e.warning}</div>` : "";
    return `<div class="plugin">
        <div class="meta">
          <div class="pname">${e.display_name} <span class="proles">${e.roles.join(", ")}</span>${pkgTag}</div>
          <div class="pdesc">${e.description || ""}</div>
          ${warn}
        </div>
        ${action}
      </div>`;
  });
  $("plugins").innerHTML = rows.join("");
}

// --- actions --------------------------------------------------------------
async function saveModels() {
  try {
    await postJSON("/set_models/", {
      detector: $("sel-detector").value,
      recognizer: $("sel-recognizer").value,
      translator: $("sel-translator").value,
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
async function clearCache() {
  if (!confirm(t("confirm.clearCache"))) return;
  try {
    const r = await postJSON("/clear_cache/", {});
    toast(t("toast.cacheCleared", { n: r.cleared }), "ok");
  } catch (e) { toast(t("toast.fail", { msg: e.message }), "err"); }
}

// --- tabs -----------------------------------------------------------------
function showView(name) {
  document.querySelectorAll(".view").forEach((el) => el.classList.toggle("active", el.dataset.view === name));
  document.querySelectorAll(".tab").forEach((el) => el.classList.toggle("active", el.dataset.view === name));
}

// --- wire up --------------------------------------------------------------
$("lang-toggle").addEventListener("click", (ev) => {
  const opt = ev.target.closest(".langopt");
  if (!opt || opt.dataset.lang === LANG) return;
  setLang(opt.dataset.lang);
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
$("clear-cache").addEventListener("click", clearCache);
$("gate-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  setToken($("gate-token").value.trim());
  load({ wrong: true });   // a still-401 after an explicit entry = wrong token
});

applyLang();
load();
