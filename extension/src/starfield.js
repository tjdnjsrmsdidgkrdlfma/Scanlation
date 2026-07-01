/* Scanlation popup — animated "Starry Night" backdrop.
 * Star particles ride a time-EVOLVING flow field (layered sines, no fixed
 * vortex centres) so the swirls keep forming and dissolving instead of locking
 * into a static rotation. Gold stars with the odd warm-white sparkle, painterly
 * trails. Self-contained, MV2 CSP-safe (packaged, no inline), dies with the
 * popup, honours prefers-reduced-motion. */
(() => {
  "use strict";
  const cvs = document.getElementById("sky");
  if (!cvs) return;
  const ctx = cvs.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  let W = 0, H = 0, dpr = 1, stars = [], t = 0, raf = 0;

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
    return true;
  }

  // Cohesive palette: mostly gold, a rare warm-white sparkle.
  function starColor() {
    const r = Math.random();
    if (r < 0.13) return "#fff4da";
    return r < 0.55 ? "#e6b450" : "#f6d488";
  }

  function seed() {
    const n = Math.max(28, Math.min(90, Math.round((W * H) / 2500)));
    stars = [];
    for (let i = 0; i < n; i++) {
      stars.push({
        x: Math.random() * W,
        y: Math.random() * H,
        r: 0.6 + Math.random() * 1.5,
        c: starColor(),
        a: 0.35 + Math.random() * 0.6,
        v: 0.4 + Math.random() * 0.6,   // per-star speed (depth)
        tw: 0.4 + Math.random() * 1.8,  // twinkle speed
        ph: Math.random() * 6.283,      // twinkle phase
      });
    }
  }

  // Direction from a layered, time-evolving angle field. Because every term
  // carries its own slow time drift, the swirl pattern morphs continuously —
  // no fixed centres, so it never settles into a static spin.
  function velocity(s) {
    const nx = s.x / W, ny = s.y / H;
    const ang =
      Math.sin(nx * 6.0 + t * 0.30) * 1.7 +
      Math.cos(ny * 5.0 - t * 0.24) * 1.7 +
      Math.sin((nx + ny) * 4.0 + t * 0.16) * 1.3 +
      Math.cos((nx - ny) * 3.2 - t * 0.11) * 1.0;
    return [Math.cos(ang) * s.v, Math.sin(ang) * s.v];
  }

  function drawStars(twinkle) {
    if (twinkle) {
      ctx.fillStyle = "rgba(7, 10, 22, 0.16)"; // fade prior frame -> brush trails
      ctx.fillRect(0, 0, W, H);
    }
    ctx.globalCompositeOperation = "lighter";
    for (const s of stars) {
      const a = twinkle
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
        const [vx, vy] = velocity(s);
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
