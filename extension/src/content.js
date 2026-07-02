/* Scanlation content script (clean-room, MV2 / Firefox, no bundler).
 * Finds images on the page, sends them to the Scanlation server, and overlays
 * the returned translation boxes. Wire contract matches the server exactly:
 *   - md5 is computed over the base64 STRING (not raw bytes)
 *   - a box is [x_min, y_min, x_max, y_max]; positioned as % of natural size
 *   - /run_ocrtsl/ is lazy(md5 only) then work(md5 + contents) on a cache miss
 */
(() => {
  "use strict";
  const ext = globalThis.browser || globalThis.chrome;
  if (window.__scanlationInjected) return;
  window.__scanlationInjected = true;

  // ----------------------------------------------------------------- md5 ----
  // Clean-room RFC 1321 over UTF-8 bytes; verified byte-equal to Python hashlib.
  function md5(str) {
    const add32 = (a, b) => (a + b) & 0xffffffff;
    const rol = (n, c) => (n << c) | (n >>> (32 - c));
    const cmn = (q, a, b, x, s, t) => add32(rol(add32(add32(a, q), add32(x, t)), s), b);
    const ff = (a, b, c, d, x, s, t) => cmn((b & c) | (~b & d), a, b, x, s, t);
    const gg = (a, b, c, d, x, s, t) => cmn((b & d) | (c & ~d), a, b, x, s, t);
    const hh = (a, b, c, d, x, s, t) => cmn(b ^ c ^ d, a, b, x, s, t);
    const ii = (a, b, c, d, x, s, t) => cmn(c ^ (b | ~d), a, b, x, s, t);
    function cycle(st, k) {
      let [a, b, c, d] = st;
      a = ff(a, b, c, d, k[0], 7, -680876936); d = ff(d, a, b, c, k[1], 12, -389564586);
      c = ff(c, d, a, b, k[2], 17, 606105819); b = ff(b, c, d, a, k[3], 22, -1044525330);
      a = ff(a, b, c, d, k[4], 7, -176418897); d = ff(d, a, b, c, k[5], 12, 1200080426);
      c = ff(c, d, a, b, k[6], 17, -1473231341); b = ff(b, c, d, a, k[7], 22, -45705983);
      a = ff(a, b, c, d, k[8], 7, 1770035416); d = ff(d, a, b, c, k[9], 12, -1958414417);
      c = ff(c, d, a, b, k[10], 17, -42063); b = ff(b, c, d, a, k[11], 22, -1990404162);
      a = ff(a, b, c, d, k[12], 7, 1804603682); d = ff(d, a, b, c, k[13], 12, -40341101);
      c = ff(c, d, a, b, k[14], 17, -1502002290); b = ff(b, c, d, a, k[15], 22, 1236535329);
      a = gg(a, b, c, d, k[1], 5, -165796510); d = gg(d, a, b, c, k[6], 9, -1069501632);
      c = gg(c, d, a, b, k[11], 14, 643717713); b = gg(b, c, d, a, k[0], 20, -373897302);
      a = gg(a, b, c, d, k[5], 5, -701558691); d = gg(d, a, b, c, k[10], 9, 38016083);
      c = gg(c, d, a, b, k[15], 14, -660478335); b = gg(b, c, d, a, k[4], 20, -405537848);
      a = gg(a, b, c, d, k[9], 5, 568446438); d = gg(d, a, b, c, k[14], 9, -1019803690);
      c = gg(c, d, a, b, k[3], 14, -187363961); b = gg(b, c, d, a, k[8], 20, 1163531501);
      a = gg(a, b, c, d, k[13], 5, -1444681467); d = gg(d, a, b, c, k[2], 9, -51403784);
      c = gg(c, d, a, b, k[7], 14, 1735328473); b = gg(b, c, d, a, k[12], 20, -1926607734);
      a = hh(a, b, c, d, k[5], 4, -378558); d = hh(d, a, b, c, k[8], 11, -2022574463);
      c = hh(c, d, a, b, k[11], 16, 1839030562); b = hh(b, c, d, a, k[14], 23, -35309556);
      a = hh(a, b, c, d, k[1], 4, -1530992060); d = hh(d, a, b, c, k[4], 11, 1272893353);
      c = hh(c, d, a, b, k[7], 16, -155497632); b = hh(b, c, d, a, k[10], 23, -1094730640);
      a = hh(a, b, c, d, k[13], 4, 681279174); d = hh(d, a, b, c, k[0], 11, -358537222);
      c = hh(c, d, a, b, k[3], 16, -722521979); b = hh(b, c, d, a, k[6], 23, 76029189);
      a = hh(a, b, c, d, k[9], 4, -640364487); d = hh(d, a, b, c, k[12], 11, -421815835);
      c = hh(c, d, a, b, k[15], 16, 530742520); b = hh(b, c, d, a, k[2], 23, -995338651);
      a = ii(a, b, c, d, k[0], 6, -198630844); d = ii(d, a, b, c, k[7], 10, 1126891415);
      c = ii(c, d, a, b, k[14], 15, -1416354905); b = ii(b, c, d, a, k[5], 21, -57434055);
      a = ii(a, b, c, d, k[12], 6, 1700485571); d = ii(d, a, b, c, k[3], 10, -1894986606);
      c = ii(c, d, a, b, k[10], 15, -1051523); b = ii(b, c, d, a, k[1], 21, -2054922799);
      a = ii(a, b, c, d, k[8], 6, 1873313359); d = ii(d, a, b, c, k[15], 10, -30611744);
      c = ii(c, d, a, b, k[6], 15, -1560198380); b = ii(b, c, d, a, k[13], 21, 1309151649);
      a = ii(a, b, c, d, k[4], 6, -145523070); d = ii(d, a, b, c, k[11], 10, -1120210379);
      c = ii(c, d, a, b, k[2], 15, 718787259); b = ii(b, c, d, a, k[9], 21, -343485551);
      st[0] = add32(st[0], a); st[1] = add32(st[1], b);
      st[2] = add32(st[2], c); st[3] = add32(st[3], d);
    }
    const bin = unescape(encodeURIComponent(str));
    const n = bin.length;
    const st = [1732584193, -271733879, -1732584194, 271733878];
    const blk = (s, off) => {
      const m = new Array(16);
      for (let j = 0; j < 16; j++) {
        m[j] = s.charCodeAt(off + j * 4) | (s.charCodeAt(off + j * 4 + 1) << 8) |
               (s.charCodeAt(off + j * 4 + 2) << 16) | (s.charCodeAt(off + j * 4 + 3) << 24);
      }
      return m;
    };
    let i;
    for (i = 64; i <= n; i += 64) cycle(st, blk(bin, i - 64));
    const tailStr = bin.substring(i - 64);
    const tail = new Array(16).fill(0);
    let k;
    for (k = 0; k < tailStr.length; k++) tail[k >> 2] |= tailStr.charCodeAt(k) << ((k % 4) * 8);
    tail[k >> 2] |= 0x80 << ((k % 4) * 8);
    if (k > 55) { cycle(st, tail); tail.fill(0); }
    tail[14] = n * 8;
    cycle(st, tail);
    const hex = (num) => {
      let s = "";
      for (let j = 0; j < 4; j++) s += ((num >> (j * 8 + 4)) & 0xf).toString(16) + ((num >> (j * 8)) & 0xf).toString(16);
      return s;
    };
    return hex(st[0]) + hex(st[1]) + hex(st[2]) + hex(st[3]);
  }

  // -------------------------------------------------------------- state ----
  const cfg = { endpoint: "http://127.0.0.1:4010", showTranslated: true, token: "", minImageDim: 80 };
  let enabled = false;
  let observer = null;
  const processed = new WeakSet();
  const processing = new WeakSet();
  const tracked = []; // { img, wrapper, boxes: [el] }

  async function loadConfig() {
    try {
      const r = await ext.storage.local.get(["endpoint", "showTranslated", "token", "minImageDim"]);
      if (r.endpoint) cfg.endpoint = r.endpoint;
      if (typeof r.showTranslated === "boolean") cfg.showTranslated = r.showTranslated;
      if (typeof r.token === "string") cfg.token = r.token;
      if (typeof r.minImageDim === "number") cfg.minImageDim = r.minImageDim;
    } catch (e) { /* storage may be unavailable in some frames */ }
  }

  // -------------------------------------------------- base64 extraction ----
  function naturalSize(el) {
    return el.tagName === "IMG" ? [el.naturalWidth, el.naturalHeight] : [el.width, el.height];
  }

  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const comma = reader.result.indexOf(",");
        resolve(reader.result.slice(comma + 1)); // strip "data:...;base64,"
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  function canvasBlob(el) {
    const [w, h] = naturalSize(el);
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    c.getContext("2d").drawImage(el, 0, 0, w, h);
    return new Promise((resolve) => c.toBlob(resolve));
  }

  async function imageToBase64(el) {
    if (el.tagName === "CANVAS") return blobToBase64(await canvasBlob(el));
    // 1) direct fetch — works same-origin or for CORS-enabled images
    try {
      const res = await fetch(el.src);
      if (res.ok) return blobToBase64(await res.blob());
    } catch (e) { /* CORS / network -> try the worker */ }
    // 2) background fetch — the extension has cross-origin host access, so it
    //    bypasses page CORS (cannot beat Referer hotlinking, e.g. pixiv i.pximg.net)
    try {
      const r = await ext.runtime.sendMessage({ type: "fetch-image", url: el.src });
      if (r && r.ok) return r.base64;
    } catch (e) { /* background unavailable -> last resort */ }
    // 3) canvas re-encode — only succeeds if the image isn't cross-origin tainted
    return blobToBase64(await canvasBlob(el));
  }

  // ------------------------------------------------------------- server ----
  async function runOcr(md5hash, base64, options) {
    const url = cfg.endpoint.replace(/\/$/, "") + "/run_ocrtsl/";
    const headers = { "Content-Type": "application/json" };
    if (cfg.token) headers["X-Auth-Token"] = cfg.token;
    const post = (body) =>
      fetch(url, { method: "POST", headers, body: JSON.stringify(body) });
    // lazy: md5 only. A cache miss returns non-2xx -> fall through to work.
    let res = await post({ md5: md5hash, options });
    if (!res.ok) res = await post({ md5: md5hash, contents: base64, options });
    if (!res.ok) {
      let detail = res.status;
      try { detail = (await res.json()).detail || detail; } catch (e) {}
      throw new Error(`server ${res.status}: ${detail}`);
    }
    return res.json();
  }

  // ------------------------------------------------------------ overlay ----
  // A bare image URL (e.g. i.pximg.net/x.jpg) renders as a browser-generated
  // ImageDocument: UA styles center + shrink-to-fit the lone <img> and toggle
  // full size on click. Once we wrap it those UA rules break — the image jumps
  // to the flow origin and our absolutely-positioned boxes scroll off-screen.
  // Detect it and take over layout so the overlay lines up on lone images too.
  function isImageDocument() {
    return typeof document.contentType === "string" && document.contentType.startsWith("image/");
  }

  function fixImageDocumentLayout(img) {
    // Save the pristine UA state first so disable()/clearAll() can restore the
    // ImageDocument's centering — otherwise the img stays display:block+margin:0
    // and jumps to the left after we unwrap it.
    img.__scanOrig = {
      cls: img.className,
      style: img.getAttribute("style") || "",
      body: document.body.getAttribute("style") || "",
    };
    // The ImageDocument UA takes the lone <img> out of normal flow (shrink-to-fit
    // + centering), which collapses our inline-block wrapper to 0 width and stacks
    // every box at one point. Restore the image to normal flow so the wrapper
    // sizes to it and the % box coords map onto the image.
    img.className = "";              // drop UA "overflowing" / "shrinkToFit"
    img.style.cursor = "default";    // was zoom-in/out
    img.style.display = "block";     // block child -> inline-block wrapper sizes to it
    img.style.position = "static";   // undo UA out-of-flow positioning
    img.style.float = "none";
    img.style.margin = "0";
    img.style.maxWidth = "100vw";    // fit viewport; boxes are % so they scale with it
    img.style.height = "auto";
    document.body.style.margin = "0";
    document.body.style.textAlign = "center"; // center the inline-block wrapper
  }

  function wrap(img) {
    const wrapper = document.createElement("span");
    wrapper.className = "scanlation-wrapper";
    img.parentNode.insertBefore(wrapper, img);
    wrapper.appendChild(img);
    if (isImageDocument()) fixImageDocumentLayout(img);
    return wrapper;
  }

  // box is [x_min, y_min, x_max, y_max] in natural px -> fractions (0-1) of the
  // natural size, so they map onto either % CSS or displayed-px equally.
  function boxFractions(box, nw, nh) {
    const [l, b, r, t] = box;
    return { left: l / nw, top: b / nh, w: (r - l) / nw, h: (t - b) / nh };
  }

  // Text follows the show-original toggle; the full text is also the hover title.
  function setBoxText(box) {
    box.textContent = cfg.showTranslated ? box.dataset.tsl : box.dataset.ocr;
    box.title = box.textContent;
  }

  function makeBox(item, nw, nh) {
    const f = boxFractions(item.box, nw, nh);
    const div = document.createElement("div");
    div.className = "scanlation-box";
    div.style.left = f.left * 100 + "%";
    div.style.top = f.top * 100 + "%";
    div.style.width = f.w * 100 + "%";
    div.style.height = f.h * 100 + "%";
    div.dataset.ocr = item.ocr;
    div.dataset.tsl = item.tsl;
    setBoxText(div);
    div.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      navigator.clipboard.writeText(div.dataset.ocr).catch(() => {});
    });
    return div;
  }

  function sizeFonts(entry) {
    const dispW = entry.img.clientWidth || entry.img.width || 1;
    const dispH = entry.img.clientHeight || entry.img.height || 1;
    const [nw, nh] = naturalSize(entry.img);
    for (const box of entry.boxes) {
      const f = boxFractions(JSON.parse(box.dataset.boxraw), nw, nh);
      const wpx = f.w * dispW;
      const hpx = f.h * dispH;
      // size font by text length vs box area so long text shrinks to fit
      const len = Math.max(1, (box.textContent || "").length);
      let fs = Math.sqrt((wpx * hpx * 0.8) / len);
      fs = Math.max(7, Math.min(fs, hpx)); // never taller than the box
      box.style.fontSize = fs + "px";
    }
  }

  function applyResult(img, result) {
    const [nw, nh] = naturalSize(img);
    if (!nw || !nh) return;
    const wrapper = wrap(img);
    const entry = { img, wrapper, boxes: [] };
    for (const item of result) {
      const box = makeBox(item, nw, nh);
      box.dataset.boxraw = JSON.stringify(item.box);
      wrapper.appendChild(box);
      entry.boxes.push(box);
    }
    tracked.push(entry);
    sizeFonts(entry);
  }

  async function processImage(img) {
    if (!enabled || processed.has(img) || processing.has(img)) return;
    const [nw, nh] = naturalSize(img);
    if (Math.min(nw, nh) < cfg.minImageDim) return; // skip icons/thin banners (shorter-side px; from /admin)
    processing.add(img);
    try {
      const base64 = await imageToBase64(img);
      const result = await runOcr(md5(base64), base64, {});
      applyResult(img, result.result || []);
      processed.add(img);
    } catch (e) {
      console.warn("[scanlation]", e.message || e);
    } finally {
      processing.delete(img);
    }
  }

  function scan(root) {
    if (root.matches && root.matches("img, canvas")) processImage(root);
    const els = root.querySelectorAll ? root.querySelectorAll("img, canvas") : [];
    els.forEach(processImage);
  }

  // -------------------------------------------------------------- toggle ---
  function retext() {
    for (const entry of tracked) {
      for (const box of entry.boxes) setBoxText(box);
      sizeFonts(entry); // text length changed -> refit
    }
  }

  function clearAll() {
    for (const entry of tracked) {
      for (const box of entry.boxes) box.remove();
      const { wrapper, img } = entry;
      if (wrapper.parentNode) {
        wrapper.parentNode.insertBefore(img, wrapper);
        wrapper.remove();
      }
      if (img.__scanOrig) {  // undo the ImageDocument layout takeover -> UA re-centers
        img.className = img.__scanOrig.cls;
        img.__scanOrig.style ? img.setAttribute("style", img.__scanOrig.style) : img.removeAttribute("style");
        img.__scanOrig.body ? document.body.setAttribute("style", img.__scanOrig.body) : document.body.removeAttribute("style");
        delete img.__scanOrig;
      }
      processed.delete(img);
    }
    tracked.length = 0;
  }

  // Tell the background page our on/off state so it can sync the page_action
  // (address-bar) icon — whether toggled from that icon or the popup.
  function reportState() {
    try { ext.runtime.sendMessage({ type: "state", enabled }); } catch (e) { /* no background */ }
  }

  function enable() {
    if (enabled) return;
    enabled = true;
    scan(document.body || document.documentElement);
    observer = new MutationObserver((muts) => {
      for (const m of muts) m.addedNodes.forEach((n) => { if (n.nodeType === 1) scan(n); });
    });
    observer.observe(document.documentElement, { childList: true, subtree: true });
    window.addEventListener("resize", onResize, { passive: true });
    reportState();
  }

  function disable() {
    if (!enabled) return;
    enabled = false;
    if (observer) { observer.disconnect(); observer = null; }
    window.removeEventListener("resize", onResize);
    clearAll();
    reportState();
  }

  let resizeTimer = null;
  function onResize() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => tracked.forEach(sizeFonts), 150);
  }

  // ------------------------------------------------------------ messages ---
  ext.runtime.onMessage.addListener((msg) => {
    switch (msg && msg.type) {
      case "enable": enable(); break;
      case "disable": disable(); break;
      case "toggle": (enabled ? disable() : enable()); break;
      case "set-endpoint": cfg.endpoint = msg.endpoint; break;
      case "set-token": cfg.token = msg.token || ""; break;
      case "set-min-image-dim": if (typeof msg.value === "number") cfg.minImageDim = msg.value; break;
      case "set-show-translated": cfg.showTranslated = !!msg.value; retext(); break;
      default: break;
    }
    return undefined;
  });

  loadConfig();
})();
