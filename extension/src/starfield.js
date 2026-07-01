/* Scanlation popup — "Starry Night" backdrop.
 *
 *   - stars sit in place and TWINKLE (slowly); they only SWAY a hair around
 *     their spot (no net drift, so nothing ever wraps in from the edges)
 *   - brighter ones show a small 4-point sparkle (✦), not a plain circle
 *   - a MILKY WAY band (dense fine dust, tapering to thin ends) + faint field stars
 * The sky is a FIXED scene built once over a generous logical height; when the
 * popup grows (#conn after Connect) we only enlarge the canvas and reveal more of
 * the same sky — no reseed. MV2 CSP-safe, prefers-reduced-motion aware. */
(() => {
  "use strict";
  const TAU = 6.2832;
  const cvs = document.getElementById("sky");
  if (!cvs) return;
  const ctx = cvs.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  let W = 0, H = 0, dpr = 1;      // actual canvas (popup) size
  let SW = 0, SH = 0;            // fixed logical sky size (built once)
  let stars = [], dust = [], band = null, baseGrad = null, seeded = false, t = 0, raf = 0;

  const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) / 1.5; // ~[-1,1]

  function starColor() {
    const r = Math.random();
    if (r < 0.13) return "#fff4da";
    return r < 0.55 ? "#e6b450" : "#f6d488";
  }

  // in-place sway params: tiny amplitude + slow speed, so a star just wobbles
  function sway() {
    return {
      ax: 0.5 + Math.random() * 1.3, ay: 0.5 + Math.random() * 1.3,
      sxs: 0.12 + Math.random() * 0.35, sys: 0.12 + Math.random() * 0.35,
      px: Math.random() * TAU, py: Math.random() * TAU,
    };
  }
  const swayX = (p) => p.x + p.s.ax * Math.sin(t * p.s.sxs + p.s.px);
  const swayY = (p) => p.y + p.s.ay * Math.sin(t * p.s.sys + p.s.py);

  function fitCanvas() {
    const w = window.innerWidth, h = window.innerHeight;
    if (w === W && h === H) return;
    W = w; H = h;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    cvs.width = Math.round(W * dpr);
    cvs.height = Math.round(H * dpr);
    cvs.style.width = W + "px";
    cvs.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function build() {
    SW = W || 304;
    SH = Math.max(H, 560);        // generous fixed height so later growth reveals, never reseeds
    baseGrad = ctx.createLinearGradient(0, 0, 0, SH);
    baseGrad.addColorStop(0, "#0c1226");
    baseGrad.addColorStop(1, "#070a16");
    const ang = -0.62;
    band = {
      cx: SW * 0.5, cy: SH * 0.5, ang,
      dx: Math.cos(ang), dy: Math.sin(ang),
      half: Math.min(SW, SH) * 0.25,
      len: Math.hypot(SW, SH) * 1.15,
    };
    seed();
    seeded = true;
  }

  function seed() {
    dust = [];
    const m = Math.max(220, Math.min(760, Math.round((SW * SH) / 230)));
    for (let i = 0; i < m; i++) {
      const along = (Math.random() - 0.5) * band.len;
      const u = along / (band.len * 0.5);                          // -1..0..1
      const perp = gauss() * band.half * Math.max(0.1, 1 - u * u); // full centre -> thin ends
      dust.push({
        x: band.cx + along * band.dx - perp * band.dy,
        y: band.cy + along * band.dy + perp * band.dx,
        r: 0.35 + Math.random() * 0.6,
        c: starColor(),
        a: 0.26 + Math.random() * 0.4,
        s: sway(),
      });
    }
    const f = Math.round(m * 0.16);                                // faint field stars off the band
    for (let i = 0; i < f; i++) {
      dust.push({
        x: Math.random() * SW, y: Math.random() * SH,
        r: 0.3 + Math.random() * 0.5,
        c: starColor(),
        a: 0.1 + Math.random() * 0.2,
        s: sway(),
      });
    }
    stars = [];
    const n = Math.max(26, Math.min(74, Math.round((SW * SH) / 3200)));
    for (let i = 0; i < n; i++) {
      const r = 0.6 + Math.random() * 1.4;
      stars.push({
        x: Math.random() * SW, y: Math.random() * SH, r,
        c: starColor(),
        a: 0.4 + Math.random() * 0.55,
        tw: 0.4 + Math.random() * 1.0,          // slow twinkle
        ph: Math.random() * TAU,
        spike: r > 1.4 || Math.random() < 0.14 ? r * (1.0 + Math.random() * 1.0) : 0,
        s: sway(),
      });
    }
  }

  function drawBandGlow() {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.translate(band.cx, band.cy);
    ctx.rotate(band.ang);
    const g = ctx.createLinearGradient(0, -band.half, 0, band.half);
    g.addColorStop(0.0, "rgba(90,110,170,0)");
    g.addColorStop(0.5, "rgba(154,154,210,0.21)");
    g.addColorStop(0.62, "rgba(210,182,126,0.15)"); // faint warm core
    g.addColorStop(1.0, "rgba(90,110,170,0)");
    ctx.fillStyle = g;
    ctx.beginPath();                            // ellipse -> the band thins toward the ends
    ctx.ellipse(0, 0, band.len / 2, band.half, 0, 0, TAU);
    ctx.fill();
    ctx.restore();
  }

  function dot(p, a) {
    ctx.globalAlpha = a;
    ctx.fillStyle = p.c;
    ctx.beginPath();
    ctx.arc(swayX(p), swayY(p), p.r, 0, TAU);
    ctx.fill();
  }

  function drawStar(s) {
    const tw = 0.35 + 0.65 * Math.abs(Math.sin(t * s.tw + s.ph));
    const a = s.a * tw;
    dot(s, a);
    if (s.spike) {
      const len = s.spike * (0.5 + 0.7 * tw);
      const w = s.r * 0.5;
      const x = swayX(s), y = swayY(s);
      ctx.globalAlpha = a * 0.55;
      ctx.beginPath();
      ctx.moveTo(x, y - len); ctx.lineTo(x - w, y);
      ctx.lineTo(x, y + len); ctx.lineTo(x + w, y); ctx.closePath(); ctx.fill();
      ctx.beginPath();
      ctx.moveTo(x - len, y); ctx.lineTo(x, y - w);
      ctx.lineTo(x + len, y); ctx.lineTo(x, y + w); ctx.closePath(); ctx.fill();
    }
  }

  function render() {
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
    ctx.fillStyle = baseGrad;
    ctx.fillRect(0, 0, SW, SH);
    drawBandGlow();
    ctx.globalCompositeOperation = "lighter";
    for (const d of dust) dot(d, d.a);
    for (const s of stars) drawStar(s);
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
  }

  function frame() {
    fitCanvas();
    if (W && !seeded) build();
    if (seeded) { render(); t += 0.016; }
    raf = requestAnimationFrame(frame);
  }

  if (reduce) {
    fitCanvas(); if (W) build(); render();
  } else {
    frame();
    window.addEventListener("unload", () => cancelAnimationFrame(raf));
  }
})();
