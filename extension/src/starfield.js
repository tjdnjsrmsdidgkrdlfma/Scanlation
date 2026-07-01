/* Scanlation popup — "Starry Night" backdrop, take 3.
 *
 *   - stars sit still and TWINKLE (slowly), drifting only a hair
 *   - the brighter ones show a small 4-point sparkle (✦), not a plain circle
 *   - the "swirl" is a MILKY WAY: a diagonal band of fine dense star-dust plus a
 *     faint band glow; the sky is sparser away from it, so the band reads
 * Gold-dominant with the odd warm-white. Self-contained, MV2 CSP-safe (packaged,
 * no inline), dies with the popup, honours prefers-reduced-motion. */
(() => {
  "use strict";
  const TAU = 6.2832;
  const cvs = document.getElementById("sky");
  if (!cvs) return;
  const ctx = cvs.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  let W = 0, H = 0, dpr = 1, baseGrad = null, t = 0, raf = 0;
  let stars = [], dust = [], band = null;

  const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) / 1.5; // ~[-1,1]

  function resize() {
    const w = window.innerWidth, h = window.innerHeight;
    if (w === W && h === H) return false;
    W = w; H = h;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    cvs.width = Math.round(W * dpr);
    cvs.height = Math.round(H * dpr);
    cvs.style.width = W + "px";
    cvs.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    baseGrad = ctx.createLinearGradient(0, 0, 0, H);
    baseGrad.addColorStop(0, "#0c1226");
    baseGrad.addColorStop(1, "#070a16");
    const ang = -0.62;                          // Milky Way tilt
    band = {
      cx: W * 0.5, cy: H * 0.5, ang,
      dx: Math.cos(ang), dy: Math.sin(ang),
      half: Math.min(W, H) * 0.27,              // band half-width
      len: Math.hypot(W, H) * 1.15,
    };
    return true;
  }

  function starColor() {
    const r = Math.random();
    if (r < 0.13) return "#fff4da";
    return r < 0.55 ? "#e6b450" : "#f6d488";
  }

  const drift = () => (Math.random() - 0.5) * 0.02;  // a hair of motion (very slow)

  function seed() {
    // fine Milky-Way dust, clustered on the band (gaussian across it)
    dust = [];
    const m = Math.max(120, Math.min(460, Math.round((W * H) / 380)));
    for (let i = 0; i < m; i++) {
      const along = (Math.random() - 0.5) * band.len;
      const perp = gauss() * band.half;
      dust.push({
        x: band.cx + along * band.dx - perp * band.dy,
        y: band.cy + along * band.dy + perp * band.dx,
        r: 0.35 + Math.random() * 0.6,
        c: starColor(),
        a: 0.22 + Math.random() * 0.36,
        vx: drift(), vy: drift(),
      });
    }
    // brighter foreground stars, scattered over the whole sky
    stars = [];
    const n = Math.max(18, Math.min(52, Math.round((W * H) / 4200)));
    for (let i = 0; i < n; i++) {
      const r = 0.6 + Math.random() * 1.4;
      stars.push({
        x: Math.random() * W, y: Math.random() * H, r,
        c: starColor(),
        a: 0.4 + Math.random() * 0.55,
        tw: 0.4 + Math.random() * 1.0,          // slow twinkle
        ph: Math.random() * TAU,
        spike: r > 1.4 || Math.random() < 0.14 ? r * (1.0 + Math.random() * 1.0) : 0,
        vx: drift(), vy: drift(),
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
    g.addColorStop(0.5, "rgba(152,152,208,0.16)");
    g.addColorStop(0.62, "rgba(208,180,124,0.12)"); // faint warm core
    g.addColorStop(1.0, "rgba(90,110,170,0)");
    ctx.fillStyle = g;
    ctx.fillRect(-band.len / 2, -band.half, band.len, band.half * 2);
    ctx.restore();
  }

  function dot(p, a) {
    ctx.globalAlpha = a;
    ctx.fillStyle = p.c;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, TAU);
    ctx.fill();
  }

  function drawStar(s) {
    const tw = 0.35 + 0.65 * Math.abs(Math.sin(t * s.tw + s.ph));
    const a = s.a * tw;
    dot(s, a);
    if (s.spike) {
      const len = s.spike * (0.5 + 0.7 * tw);
      const w = s.r * 0.5;
      ctx.globalAlpha = a * 0.55;
      ctx.beginPath();
      ctx.moveTo(s.x, s.y - len); ctx.lineTo(s.x - w, s.y);
      ctx.lineTo(s.x, s.y + len); ctx.lineTo(s.x + w, s.y); ctx.closePath(); ctx.fill();
      ctx.beginPath();
      ctx.moveTo(s.x - len, s.y); ctx.lineTo(s.x, s.y - w);
      ctx.lineTo(s.x + len, s.y); ctx.lineTo(s.x, s.y + w); ctx.closePath(); ctx.fill();
    }
  }

  function render() {
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
    ctx.fillStyle = baseGrad;
    ctx.fillRect(0, 0, W, H);
    drawBandGlow();
    ctx.globalCompositeOperation = "lighter";
    for (const d of dust) dot(d, d.a);
    for (const s of stars) drawStar(s);
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
  }

  function step() {
    for (const p of dust) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < -4) p.x = W + 4; else if (p.x > W + 4) p.x = -4;
      if (p.y < -4) p.y = H + 4; else if (p.y > H + 4) p.y = -4;
    }
    for (const s of stars) {
      s.x += s.vx; s.y += s.vy;
      if (s.x < -4) s.x = W + 4; else if (s.x > W + 4) s.x = -4;
      if (s.y < -4) s.y = H + 4; else if (s.y > H + 4) s.y = -4;
    }
  }

  function frame() {
    if (resize()) seed();
    if (W && H) { render(); step(); t += 0.016; }
    raf = requestAnimationFrame(frame);
  }

  if (reduce) {
    resize(); seed(); render();
  } else {
    frame();
    window.addEventListener("unload", () => cancelAnimationFrame(raf));
  }
})();
