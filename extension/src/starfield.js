/* Scanlation popup — "Starry Night" backdrop, take 2.
 *
 * Realism first: real stars don't race around — they sit still and TWINKLE.
 * So the swirl belongs to the SKY, not the stars:
 *   - a slow, faint nebula (a few big soft glows drifting) = the atmospheric swirl
 *   - stars are fixed points that twinkle; the brighter ones show a 4-point
 *     sparkle (✦), not a plain circle
 * Gold-dominant with the odd warm-white. Self-contained, MV2 CSP-safe (packaged,
 * no inline), dies with the popup, honours prefers-reduced-motion. */
(() => {
  "use strict";
  const TAU = 6.2832;
  const cvs = document.getElementById("sky");
  if (!cvs) return;
  const ctx = cvs.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  let W = 0, H = 0, dpr = 1, stars = [], baseGrad = null, t = 0, raf = 0;

  // Slow atmospheric glows — the only thing that actually moves (gently).
  const NEBULA = [
    { bx: 0.30, by: 0.28, ax: 0.10, ay: 0.07, sx: 0.16, sy: 0.12, ph: 0.0, R: 0.80, col: "70,95,175" },
    { bx: 0.73, by: 0.66, ax: 0.09, ay: 0.10, sx: 0.12, sy: 0.18, ph: 2.1, R: 0.72, col: "150,112,56" },
    { bx: 0.52, by: 0.88, ax: 0.08, ay: 0.06, sx: 0.10, sy: 0.14, ph: 4.0, R: 0.60, col: "58,80,150" },
  ];

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
    return true;
  }

  function starColor() {
    const r = Math.random();
    if (r < 0.13) return "#fff4da";
    return r < 0.55 ? "#e6b450" : "#f6d488";
  }

  function seed() {
    const n = Math.max(26, Math.min(78, Math.round((W * H) / 3000)));
    stars = [];
    for (let i = 0; i < n; i++) {
      const r = 0.5 + Math.random() * 1.4;
      stars.push({
        x: Math.random() * W,
        y: Math.random() * H,
        r,
        c: starColor(),
        a: 0.35 + Math.random() * 0.6,
        tw: 0.5 + Math.random() * 2.2,          // twinkle speed
        ph: Math.random() * TAU,                // twinkle phase
        // brighter stars get a sparkle; most stay tiny points
        spike: r > 1.15 || Math.random() < 0.22 ? r * (2.4 + Math.random() * 2.2) : 0,
      });
    }
  }

  function drawNebula() {
    ctx.globalCompositeOperation = "lighter";
    const maxR = Math.max(W, H);
    for (const b of NEBULA) {
      const cx = (b.bx + b.ax * Math.sin(t * b.sx + b.ph)) * W;
      const cy = (b.by + b.ay * Math.cos(t * b.sy + b.ph)) * H;
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, b.R * maxR);
      g.addColorStop(0, `rgba(${b.col},0.11)`);
      g.addColorStop(1, `rgba(${b.col},0)`);
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, W, H);
    }
  }

  function drawStar(s) {
    const tw = 0.35 + 0.65 * Math.abs(Math.sin(t * s.tw + s.ph));
    const a = s.a * tw;
    ctx.fillStyle = s.c;
    ctx.globalAlpha = a;
    ctx.beginPath();
    ctx.arc(s.x, s.y, s.r, 0, TAU);
    ctx.fill();
    if (s.spike) {
      const len = s.spike * (0.45 + 0.9 * tw); // spikes pulse with the twinkle
      const w = s.r * 0.6;
      ctx.globalAlpha = a * 0.6;
      ctx.beginPath();                          // vertical
      ctx.moveTo(s.x, s.y - len); ctx.lineTo(s.x - w, s.y);
      ctx.lineTo(s.x, s.y + len); ctx.lineTo(s.x + w, s.y); ctx.closePath(); ctx.fill();
      ctx.beginPath();                          // horizontal
      ctx.moveTo(s.x - len, s.y); ctx.lineTo(s.x, s.y - w);
      ctx.lineTo(s.x + len, s.y); ctx.lineTo(s.x, s.y + w); ctx.closePath(); ctx.fill();
    }
  }

  function render() {
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
    ctx.fillStyle = baseGrad;
    ctx.fillRect(0, 0, W, H);
    drawNebula();
    ctx.globalCompositeOperation = "lighter";
    for (const s of stars) drawStar(s);
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
  }

  function frame() {
    if (resize()) seed();
    if (W && H) { render(); t += 0.016; }
    raf = requestAnimationFrame(frame);
  }

  if (reduce) {
    resize(); seed(); render();
  } else {
    frame();
    window.addEventListener("unload", () => cancelAnimationFrame(raf));
  }
})();
