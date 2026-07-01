/* Scanlation popup — animated "Starry Night" backdrop.
 * Star particles drift along a swirl vector field (a couple of Van Gogh-style
 * vortices) and leave painterly trails; each twinkles in gold / night-blue.
 * Self-contained, no imports (MV2 CSP: packaged script, no inline). Dies with
 * the popup. Honours prefers-reduced-motion (static field, no loop). */
(() => {
  "use strict";
  const cvs = document.getElementById("sky");
  if (!cvs) return;
  const ctx = cvs.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const PALETTE = ["#f6d488", "#e6b450", "#cfe0ff", "#8fa8ef", "#ffffff"];
  let W = 0, H = 0, dpr = 1, stars = [], centers = [], t = 0, raf = 0;

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
    // two counter-rotating vortices, like the swirls in Starry Night
    centers = [
      { x: W * 0.28, y: H * 0.24, s: 30 },
      { x: W * 0.74, y: H * 0.62, s: -24 },
    ];
    return true;
  }

  function seed() {
    const n = Math.max(28, Math.min(96, Math.round((W * H) / 2500)));
    stars = [];
    for (let i = 0; i < n; i++) {
      stars.push({
        x: Math.random() * W,
        y: Math.random() * H,
        r: 0.6 + Math.random() * 1.6,
        c: PALETTE[(Math.random() * PALETTE.length) | 0],
        a: 0.35 + Math.random() * 0.6,
        tw: 0.4 + Math.random() * 2.0,   // twinkle speed
        ph: Math.random() * 6.283,       // twinkle phase
      });
    }
  }

  // Velocity from the swirl field: tangential rotation around each vortex,
  // softened near the core and speed-capped so motion stays gentle.
  function velocity(x, y) {
    let vx = 0, vy = 0;
    for (const c of centers) {
      const dx = x - c.x, dy = y - c.y;
      const f = c.s / (dx * dx + dy * dy + 700);
      vx += -dy * f;
      vy += dx * f;
    }
    vy -= 0.02; // faint upward drift
    const sp = Math.hypot(vx, vy);
    if (sp > 1.1) { vx *= 1.1 / sp; vy *= 1.1 / sp; }
    return [vx, vy];
  }

  function drawStars(trail) {
    if (trail) {
      ctx.fillStyle = "rgba(7, 10, 22, 0.16)"; // fade prior frame -> brush trails
      ctx.fillRect(0, 0, W, H);
    }
    ctx.globalCompositeOperation = "lighter";
    for (const s of stars) {
      const a = trail
        ? s.a * (0.32 + 0.68 * Math.abs(Math.sin(t * s.tw + s.ph)))
        : s.a;
      ctx.globalAlpha = a;
      ctx.fillStyle = s.c;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, 6.2832);
      ctx.fill();
    }
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
  }

  function frame() {
    if (resize()) seed();
    if (W && H) {
      drawStars(true);
      for (const s of stars) {
        const [vx, vy] = velocity(s.x, s.y);
        s.x += vx; s.y += vy;
        if (s.x < -4) s.x = W + 4; else if (s.x > W + 4) s.x = -4;
        if (s.y < -4) s.y = H + 4; else if (s.y > H + 4) s.y = -4;
      }
      t += 0.016;
    }
    raf = requestAnimationFrame(frame);
  }

  if (reduce) {
    resize(); seed();
    ctx.fillStyle = "rgba(7, 10, 22, 1)";
    ctx.fillRect(0, 0, W, H);
    drawStars(false);
  } else {
    frame();
    window.addEventListener("unload", () => cancelAnimationFrame(raf));
  }
})();
