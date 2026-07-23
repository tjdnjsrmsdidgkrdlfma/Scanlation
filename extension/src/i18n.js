"use strict";
// Shared i18n for the extension's user-facing text: the popup UI and the content
// script's failure badge. A classic script like constants.js/util.js — published
// on globalThis for every realm. The chosen language lives in ext.storage.local
// (key "lang", default "en" to match the admin SPA), so all three realms share
// one setting; each realm calls setLang() once it has read storage. `{x}` tokens
// are filled by t(key, {x: ...}). Mirrors the admin SPA's i18n.js table.
globalThis.SCANI18N = (() => {
  const STRINGS = {
    en: {
      "brand.tagline": "manga OCR + translation overlay",
      "field.server": "Server",
      "field.token": "Token",
      "field.token.opt": "(optional)",
      "ph.token": "X-Auth-Token (leave blank if none)",
      "btn.connect": "Connect",
      "rule.languages": "languages",
      "rule.engines": "engines",
      "field.from": "From",
      "field.to": "To",
      "field.detector": "Detector",
      "field.recognizer": "Recognizer",
      "field.translator": "Translator",
      "btn.translate": "Translate",
      "btn.clear": "Clear",
      "chk.showOriginal": "Show original",
      "status.connecting": "connecting…",
      "status.connected": "connected · v{v}",
      "status.authFail": "auth failed — check the token",
      "status.unreachable": "cannot reach server: {msg}",
      "badge.fail": "Translation failed",
      "title.on": "Scanlation: translating — click to stop",
      "title.off": "Scanlation: click to translate this tab",
    },
    ko: {
      "brand.tagline": "만화 OCR + 번역 오버레이",
      "field.server": "서버",
      "field.token": "토큰",
      "field.token.opt": "(선택)",
      "ph.token": "X-Auth-Token (없으면 비워두세요)",
      "btn.connect": "접속",
      "rule.languages": "언어",
      "rule.engines": "엔진",
      "field.from": "원문",
      "field.to": "번역",
      "field.detector": "검출기",
      "field.recognizer": "인식기",
      "field.translator": "번역기",
      "btn.translate": "번역",
      "btn.clear": "지우기",
      "chk.showOriginal": "원문 보기",
      "status.connecting": "접속 중…",
      "status.connected": "접속됨 · v{v}",
      "status.authFail": "인증 실패 — 토큰을 확인하세요",
      "status.unreachable": "서버에 접속할 수 없음: {msg}",
      "badge.fail": "번역 실패",
      "title.on": "Scanlation: 번역 중 — 클릭하면 중지",
      "title.off": "Scanlation: 클릭하면 이 탭 번역",
    },
  };
  const api = {
    STRINGS,
    lang: "en", // overridden once a realm reads storage (see setLang)
    normalize(l) { return l === "ko" || l === "en" ? l : "en"; },
    setLang(l) { api.lang = api.normalize(l); return api.lang; },
    t(key, vars) {
      const table = STRINGS[api.lang] || STRINGS.en;
      let s = table[key] || STRINGS.en[key] || key;
      if (vars) for (const k in vars) s = s.split("{" + k + "}").join(vars[k]);
      return s;
    },
  };
  return api;
})();
