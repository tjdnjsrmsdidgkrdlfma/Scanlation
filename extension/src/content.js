/* Scanlation content script (clean-room, MV2 / Firefox, no bundler).
 * Finds images on the page, sends them to the Scanlation server, and overlays
 * the returned translation boxes. Wire contract matches the server exactly:
 *   - md5 is computed over the base64 STRING (not raw bytes)
 *   - a box is [x_min, y_min, x_max, y_max]; positioned as % of natural size
 *   - /run_pipeline/ is lazy(md5 only) then work(md5 + contents) on a cache miss
 */
(() => {
  "use strict";
  const ext = globalThis.browser || globalThis.chrome;
  if (window.__scanlationInjected) return;
  window.__scanlationInjected = true;

  // -------------------------------------------------------------- state ----
  const cfg = {
    endpoint: globalThis.SCAN.ENDPOINT,
    showTranslated: globalThis.SCAN.SHOW_TRANSLATED,
    token: "",
    minImageDim: globalThis.SCAN.MIN_IMAGE_DIM,
  };
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
  async function runPipeline(md5hash, base64, options) {
    const base = cfg.endpoint.replace(/\/$/, "");
    const headers = { "Content-Type": "application/json" };
    if (cfg.token) headers["X-Auth-Token"] = cfg.token;
    const post = (path, body) =>
      fetch(base + path, { method: "POST", headers, body: JSON.stringify(body) });
    // 1) cache probe (md5 only, no image upload) -> always 200 {result: cached|null}.
    let res = await post("/run_lookup/", { md5: md5hash, options });
    let body = res.ok ? await res.json() : null;
    // 2) miss (result null, or probe unreachable) -> real work with the image bytes.
    if (!body || body.result == null) {
      res = await post("/run_pipeline/", { md5: md5hash, contents: base64, options });
      if (!res.ok) {
        let detail = res.status;
        try { detail = (await res.json()).detail || detail; } catch (e) {}
        throw new Error(`server ${res.status}: ${detail}`);
      }
      body = await res.json();
    }
    return body;
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
    // The UA centered the lone image both ways (margin:auto + inset:0). Match
    // that so enabling the overlay doesn't shift the image: the earlier fix
    // restored horizontal centering (textAlign) but not vertical, so short
    // images jumped up to the flow top.
    document.body.style.margin = "0";
    document.body.style.minHeight = "100vh";
    document.body.style.display = "flex";
    document.body.style.justifyContent = "center"; // horizontal
    document.body.style.alignItems = "center";     // vertical
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
    box.textContent = cfg.showTranslated ? box.dataset.destination : box.dataset.source;
    box.title = box.textContent;
  }

  function makeBox(item, nw, nh) {
    const f = boxFractions(item.bounds, nw, nh);
    const div = document.createElement("div");
    div.className = "scanlation-box";
    div.style.left = f.left * 100 + "%";
    div.style.top = f.top * 100 + "%";
    div.style.width = f.w * 100 + "%";
    div.style.height = f.h * 100 + "%";
    div.dataset.source = item.source;
    div.dataset.destination = item.destination;
    setBoxText(div);
    div.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      navigator.clipboard.writeText(div.dataset.source).catch(() => {});
    });
    return div;
  }

  function sizeFonts(entry) {
    const dispW = entry.img.clientWidth || entry.img.width || 1;
    const dispH = entry.img.clientHeight || entry.img.height || 1;
    const [nw, nh] = naturalSize(entry.img);
    for (const box of entry.boxes) {
      const f = boxFractions(JSON.parse(box.dataset.boundsraw), nw, nh);
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
      box.dataset.boundsraw = JSON.stringify(item.bounds);
      wrapper.appendChild(box);
      entry.boxes.push(box);
    }
    tracked.push(entry);
    sizeFonts(entry);
  }

  // A whole-request failure has no box coords, so pin a status chip to the image
  // corner instead of a text box. Reuse wrap()/tracked so clearAll() cleans it up.
  function showError(img, msg) {
    const [nw, nh] = naturalSize(img);
    if (!nw || !nh) return;
    const wrapper = wrap(img);
    const badge = document.createElement("div");
    badge.className = "scanlation-badge scanlation-error";
    badge.textContent = globalThis.SCAN.MSG_FAIL;
    badge.title = msg; // cause (e.g. "server 502: ...") on hover
    wrapper.appendChild(badge);
    // boxes stays empty: the badge is a child of wrapper (removed with it) and
    // has no boundsraw, so it must not go through sizeFonts()/onResize().
    tracked.push({ img, wrapper, boxes: [] });
  }

  async function processImage(img) {
    if (!enabled || processed.has(img) || processing.has(img)) return;
    const [nw, nh] = naturalSize(img);
    if (Math.min(nw, nh) < cfg.minImageDim) return; // skip icons/thin banners (shorter-side px; from /admin)
    processing.add(img);
    img.classList.add("scanlation-loading"); // faint blur so an in-flight request is visible
    try {
      const base64 = await imageToBase64(img);
      const result = await runPipeline(md5(base64), base64, {});
      if (!enabled) return; // disabled mid-flight -> drop the late result (finally clears the blur)
      applyResult(img, result.result || []);
      processed.add(img);
    } catch (e) {
      console.warn("[scanlation]", e.message || e);
      if (enabled) {
        showError(img, e.message || String(e));
        processed.add(img); // don't let a rescan re-process (would double-wrap); retry via Clear->Translate
      }
    } finally {
      processing.delete(img);
      img.classList.remove("scanlation-loading");
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
    // In-flight images aren't in `tracked` yet, so strip any lingering blur here
    // too — otherwise a Clear during processing leaves them dimmed until the
    // request settles (their finally clears it, but not until then).
    document.querySelectorAll(".scanlation-loading")
      .forEach((el) => el.classList.remove("scanlation-loading"));
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
