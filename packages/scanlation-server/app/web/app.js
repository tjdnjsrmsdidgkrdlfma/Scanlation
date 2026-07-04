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
    "models.device": "연산 장치",
    "models.device.desc": "검출·인식을 어느 장치에서 돌릴지 정합니다. 저장하면 모델을 새 장치로 다시 로드합니다. GPU는 VRAM 여유가 있을 때만 고르세요(LLM과 별개).",
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
    "opt.rtdetr.conf": "신뢰도 임계값 — 높이면 약한/노이즈 박스 제거",
    "opt.rtdetr.nms_iou": "겹침(IoU)이 이 값↑인 박스 제거 (1.0 = 끔)",
    "opt.rtdetr.contain_thresh": "작은 박스가 큰 박스에 이 비율↑ 포함되면 제거 (IoS; 1.0 = 끔)",
    "opt.ollama.model": "ollama 모델 태그 (예: gemma4:31b). 필수 — /admin에서 선택",
    "opt.ollama.num_ctx": "KV 캐시 컨텍스트 창 (번역 입력은 짧음)",
    "opt.ollama.num_gpu": "GPU로 오프로드할 레이어 수",
    "opt.llamacpp.model": "모델 id (서버 /v1/models). 필수 — /admin에서 선택",
    "opt.llamacpp.max_tokens": "생성 최대 토큰 수",
    "plugins.h2": "플러그인 설치",
    "plugins.hint": "패키지 + 가중치 원클릭 설치",
    "plugins.installed": "설치됨",
    "plugins.install": "설치",
    "plugins.installing": "설치 중…",
    "plugins.queued": "대기 중…",
    "plugins.details": "자세히",
    "plugins.phase.package": "패키지 설치 중…",
    "plugins.phase.weights": "가중치 다운로드 중…",
    "plugins.phase.done": "완료",
    "tab.maintenance": "유지보수",
    "maint.h2": "유지보수",
    "maint.hint": "캐시 관리",
    "maint.cache.title": "캐시 비우기",
    "maint.cache.desc": "저장된 모든 캐시(페이지 결과 + 번역 기록)를 지워 다음 접속 때 전 과정(검출·인식·번역)을 새로 실행합니다.",
    "maint.cache.btn": "캐시 비우기",
    "tab.behavior": "동작",
    "behavior.h2": "동작",
    "behavior.hint": "확장 동작 설정",
    "behavior.minDim.label": "최소 이미지 변 (px)",
    "behavior.minDim.desc": "이미지의 짧은 변이 이 값보다 작으면 아이콘·배너로 보고 번역하지 않습니다. 확장이 접속 시 이 값을 받아 적용합니다. 0 = 모든 이미지 번역.",
    "toast.behaviorSaved": "동작 설정 저장됨",
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
    "toast.installed": "{name} 설치 완료",
    "toast.installFail": "{name} 설치 실패: {msg}",
    "gate.tokenPh": "접근 토큰",
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
    "models.device": "Compute device",
    "models.device.desc": "Which device runs detection + recognition. Saving reloads the models on the new device. Pick GPU only when VRAM is free (separate from the LLM).",
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
    "plugins.installing": "Installing…",
    "plugins.queued": "Queued…",
    "plugins.details": "Details",
    "plugins.phase.package": "Installing package…",
    "plugins.phase.weights": "Downloading weights…",
    "plugins.phase.done": "Done",
    "tab.maintenance": "Maintenance",
    "maint.h2": "Maintenance",
    "maint.hint": "Cache management",
    "maint.cache.title": "Clear cache",
    "maint.cache.desc": "Removes all cached data (page results + translation log) so the full pipeline (detect · recognize · translate) re-runs next time.",
    "maint.cache.btn": "Clear cache",
    "tab.behavior": "Behavior",
    "behavior.h2": "Behavior",
    "behavior.hint": "Extension behavior settings",
    "behavior.minDim.label": "Min image side (px)",
    "behavior.minDim.desc": "Images whose shorter side is under this are treated as icons/banners and not translated. The extension picks this up on connect. 0 = translate everything.",
    "toast.behaviorSaved": "Behavior settings saved",
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
    "toast.installed": "{name} installed",
    "toast.installFail": "{name} install failed: {msg}",
    "gate.tokenPh": "Access token",
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
  renderBehavior();
  syncInstallPoll();
}

// While the server reports an install running that THIS tab isn't driving (e.g. a
// page reloaded mid-install, or another tab kicked it off), poll so the row flips
// from "설치 중…" to "설치됨" on its own. The driving tab updates via its stream.
let installPoll = null;
function syncInstallPoll() {
  const serverBusy = ((DATA && DATA.installing) || []).length > 0;
  const drivingHere = currentInstall != null || installQueue.length > 0;
  if (serverBusy && !drivingHere) {
    if (!installPoll) installPoll = setInterval(pollInstalls, 2500);
  } else if (installPoll) {
    clearInterval(installPoll);
    installPoll = null;
  }
}
async function pollInstalls() {
  if (currentInstall != null || installQueue.length) return;  // a stream took over
  let snap;
  try { snap = await api("/get_settings/"); } catch (_) { return; }
  const wasBusy = (DATA.installing || []).length;
  DATA = snap;
  const nowBusy = (DATA.installing || []).length;
  if (wasBusy && !nowBusy) render();   // an install finished -> full refresh (new engines in dropdowns)
  else { renderPlugins(); syncInstallPoll(); }  // still going -> just refresh the rows
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
  $("sel-device").value = DATA.selection.device;
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
  // Installs the server says are running now (a background thread outlives the
  // streaming request) show as "설치 중…" even on a fresh page load / other tab.
  const serverInstalling = new Set(DATA.installing || []);
  const rows = Object.values(byName).map((e) => {
    // One-shot install: the button pip-installs the package (if missing) AND
    // downloads the weights in a single action; the live log shows both phases as
    // they run. `installed` (weights present, which implies the package) is the
    // only "fully done" state, so it's just Install ↔ Installed — same footprint.
    const action = e.installed
      ? `<span class="pa pa-done"><span class="pa-ico">✓</span>${t("plugins.installed")}</span>`
      : serverInstalling.has(e.name)
      ? `<span class="pa pa-busy"><span class="spin"></span>${t("plugins.installing")}</span>`
      : `<button class="pa pa-install" data-install="${e.name}">${t("plugins.install")}</button>`;
    const warn = e.warning ? `<div class="pwarn">⚠ ${e.warning}</div>` : "";
    return `<div class="plugin" data-name="${e.name}">
        <div class="meta">
          <div class="pname">${e.display_name} <span class="proles">${e.roles.join(", ")}</span></div>
          <div class="pdesc">${e.description || ""}</div>
          ${warn}
        </div>
        ${action}
        <div class="plog" hidden></div>
      </div>`;
  });
  $("plugins").innerHTML = rows.join("");
  // If the list re-renders mid-queue (e.g. a language toggle), re-apply the
  // in-flight/queued chips so a busy row doesn't fall back to a plain "설치".
  if (currentInstall) setAction(currentInstall, busyChip());
  for (const n of installQueue) setAction(n, queuedChip());
}

// --- behavior (client settings, delivered to the extension via handshake) -
function renderBehavior() {
  $("min-image-dim").value = DATA.selection.min_image_dim;
}

// --- actions --------------------------------------------------------------
async function saveModels() {
  try {
    await postJSON("/set_engines/", {
      detector: $("sel-detector").value,
      recognizer: $("sel-recognizer").value,
      translator: $("sel-translator").value,
    });
    await postJSON("/set_languages/", { lang_src: $("sel-lang_src").value, lang_dst: $("sel-lang_dst").value });
    await postJSON("/set_device/", { device: $("sel-device").value });
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
// Installs run one at a time but you can queue several: clicking more plugins
// while one runs marks them "대기 중…" and they install back-to-back (serial by
// design — concurrent pip into the same --target, and the process-global stdout
// capture, can't safely overlap). The queue drains, then one load() syncs the list.
const installQueue = [];   // names waiting to install, in click order
let currentInstall = null; // the name currently installing (or null when idle)

function enqueueInstall(name) {
  if (name === currentInstall || installQueue.includes(name)) return;  // already going/queued
  installQueue.push(name);
  if (currentInstall) setAction(name, queuedChip());  // something's running -> show as waiting
  pumpQueue();
}
async function pumpQueue() {
  if (currentInstall) return;                 // one at a time
  const name = installQueue.shift();
  if (!name) { await load(); return; }        // queue drained -> reconcile the list with the server
  currentInstall = name;
  await runInstall(name);
  currentInstall = null;
  pumpQueue();                                // next, or the final load() when empty
}

// Install one plugin: swap its action to the spinner chip, open the progress panel
// (phase label + bar; raw log hidden behind "자세히"), and stream into it. Success ->
// show "설치됨" in place (the final load() reconciles); failure -> keep the log open
// (error visible) and restore the Install button so it can be retried.
async function runInstall(name) {
  setAction(name, busyChip());
  const row = document.querySelector(`.plugin[data-name="${name}"]`);
  const log = openPluginLog(row);
  let ok = false;
  try {
    await streamInstall(name, (ev) => log.handle(ev));
    ok = true;
  } catch (e) {
    log.fail(e.message);
    toast(t("toast.installFail", { name, msg: e.message }), "err");
  }
  if (ok) {
    toast(t("toast.installed", { name }), "ok");
    setAction(name, doneChip());
    hidePluginLog(name);
  } else {
    setAction(name, installButton(name));      // keep the (error) log; allow retry
  }
}

// --- action chip builders (one .pa footprint for every state) --------------
function chip(variant, text) {
  const s = document.createElement("span");
  s.className = "pa " + variant;
  s.textContent = text;
  return s;
}
function installButton(name) {
  const b = document.createElement("button");
  b.className = "pa pa-install";
  b.dataset.install = name;
  b.textContent = t("plugins.install");
  return b;
}
function queuedChip() { return chip("pa-queued", t("plugins.queued")); }
function doneChip() {
  const el = chip("pa-done", t("plugins.installed"));
  el.insertAdjacentHTML("afterbegin", `<span class="pa-ico">✓</span>`);
  return el;
}
function busyChip() {
  const el = chip("pa-busy", t("plugins.installing"));
  el.insertAdjacentHTML("afterbegin", `<span class="spin"></span>`);
  return el;
}
// Replace a row's current action element (any .pa) with a new one, leaving the
// meta + log panel untouched.
function setAction(name, el) {
  const row = document.querySelector(`.plugin[data-name="${name}"]`);
  if (!row) return;
  const cur = row.querySelector(".pa");
  if (cur) cur.replaceWith(el);
  else row.querySelector(".plog").before(el);
}
function hidePluginLog(name) {
  const row = document.querySelector(`.plugin[data-name="${name}"]`);
  const p = row && row.querySelector(".plog");
  if (p) { p.hidden = true; p.innerHTML = ""; }
}

// POST the install and read the NDJSON event stream line by line (fetch + reader,
// so the X-Auth-Token header still applies — EventSource can't send headers). Each
// line is one JSON event; an `error` event makes this throw after the stream ends.
async function streamInstall(name, onEvent) {
  const tok = getToken();
  const res = await fetch("/install_plugin_stream/", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(tok ? { "X-Auth-Token": tok } : {}) },
    body: JSON.stringify({ name }),
  });
  if (!res.ok || !res.body) {
    let detail = `HTTP ${res.status}`;
    try { const b = await res.json(); if (b && b.detail) detail = b.detail; } catch (_) { /* non-json */ }
    throw new Error(detail);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "", errMsg = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      let ev;
      try { ev = JSON.parse(line); } catch (_) { continue; }  // skip a partial/garbled line
      if (ev.event === "error") errMsg = ev.message;
      onEvent(ev);
    }
  }
  if (errMsg) throw new Error(errMsg);
}

// Build the progress panel under a plugin row and return a controller. The default
// view is just a phase label + a progress bar (indeterminate during pip, which has
// no overall %; determinate once a weights-download % arrives). The raw log is kept
// hidden behind a "자세히" toggle, and auto-revealed on failure.
function openPluginLog(row) {
  const panel = row.querySelector(".plog");
  panel.hidden = false;
  panel.innerHTML =
    `<div class="plog-head">` +
      `<span class="plog-phase"></span>` +
      `<span class="plog-pct"></span>` +
      `<button type="button" class="plog-toggle" hidden>${t("plugins.details")}</button>` +
    `</div>` +
    `<div class="plog-bar indet"><i></i></div>` +
    `<div class="plog-out" hidden></div>`;
  const phaseEl = panel.querySelector(".plog-phase");
  const pctEl = panel.querySelector(".plog-pct");
  const bar = panel.querySelector(".plog-bar");
  const fill = panel.querySelector(".plog-bar i");
  const out = panel.querySelector(".plog-out");
  const toggle = panel.querySelector(".plog-toggle");
  let live = null;  // single element mirroring the current \r progress line in the log

  toggle.addEventListener("click", () => {
    out.hidden = !out.hidden;
    toggle.classList.toggle("open", !out.hidden);
    if (!out.hidden) out.scrollTop = out.scrollHeight;
  });
  // indeterminate = no overall % (pip phase): animate the bar and hide the number
  const setIndet = (on) => { bar.classList.toggle("indet", on); if (on) { fill.style.width = ""; pctEl.textContent = ""; } };
  const record = (line, kind) => {  // every line goes to the (hidden) details log
    toggle.hidden = false;
    if (kind === "progress") {
      if (!live) { live = document.createElement("div"); live.className = "plog-line plog-progress"; out.appendChild(live); }
      live.textContent = line;
    } else {
      const d = document.createElement("div");
      d.className = "plog-line" + (kind === "err" ? " plog-err" : "");
      d.textContent = line;
      out.appendChild(d);
      live = null;
    }
    out.scrollTop = out.scrollHeight;
  };

  return {
    handle(ev) {
      if (ev.event === "phase") {
        phaseEl.textContent = t("plugins.phase." + ev.phase);
        setIndet(true);             // new phase -> unknown progress until a % arrives
      } else if (ev.event === "log") {
        const line = stripAnsi(ev.line);
        if (!line) return;
        // huggingface_hub prints its download bars as \n lines (not \r), so parse
        // the % from any line — not just `progress` ones. Skip tqdm's "Fetching N
        // files" file-count bar so the visible % tracks bytes, not file count.
        if (!/^Fetching /.test(line)) {
          const pct = parsePct(line);
          if (pct != null) { setIndet(false); fill.style.width = pct + "%"; pctEl.textContent = Math.round(pct) + "%"; }
        }
        // Collapse tqdm bar refreshes ("…: 45%|██…|") onto one live log line;
        // append everything else. Keeps the hidden log short during a big download.
        record(line, (/%\s*\|/.test(line) || ev.stream === "progress") ? "progress" : "log");
      } else if (ev.event === "done") {
        phaseEl.textContent = t("plugins.phase.done");
        setIndet(false); fill.style.width = "100%"; pctEl.textContent = "";
      }
      // `error` events are surfaced by streamInstall's throw -> fail() below.
    },
    fail(msg) {
      phaseEl.textContent = "✖ " + (msg || "error");
      pctEl.textContent = "";
      panel.querySelector(".plog-head").classList.add("err");
      bar.classList.remove("indet"); bar.classList.add("err"); fill.style.width = "100%";
      record("✖ " + (msg || "error"), "err");
      out.hidden = false;           // reveal the log so the cause is visible
      toggle.classList.add("open");
    },
  };
}

// Percent from a progress line like "model.safetensors 63%|██| 100M/158M", or null.
function parsePct(s) {
  const m = /(\d{1,3}(?:\.\d+)?)\s*%/.exec(s);
  if (!m) return null;
  const v = parseFloat(m[1]);
  return v >= 0 && v <= 100 ? v : null;
}
// Drop ANSI escape sequences (tqdm uses \x1b[A cursor moves + colours) and trim.
function stripAnsi(s) { return s.replace(/\x1b\[[0-9;]*[A-Za-z]/g, "").trim(); }
async function clearCache() {
  if (!confirm(t("confirm.clearCache"))) return;
  try {
    const r = await postJSON("/clear_cache/", {});
    toast(t("toast.cacheCleared", { n: r.cleared }), "ok");
  } catch (e) { toast(t("toast.fail", { msg: e.message }), "err"); }
}
async function saveBehavior() {
  try {
    const n = parseInt($("min-image-dim").value, 10);
    await postJSON("/set_client_config/", { min_image_dim: Number.isFinite(n) ? n : 0 });
    toast(t("toast.behaviorSaved"), "ok");
    await load();
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
  if (btn) enqueueInstall(btn.dataset.install);
});
$("clear-cache").addEventListener("click", clearCache);
$("save-behavior").addEventListener("click", saveBehavior);
$("gate-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  setToken($("gate-token").value.trim());
  load({ wrong: true });   // a still-401 after an explicit entry = wrong token
});

applyLang();
// No token yet? Show the login scene immediately (before the get_settings round
// trip) so the app UI never flashes first. A valid stored token skips straight
// to load(); auth-off servers just hide the gate the moment load() succeeds.
if (!getToken()) showGate(false);
load();
