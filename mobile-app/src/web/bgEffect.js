// Web-only animated "constellation" background (nodes + connecting lines),
// matching the reference hero. Lazy-loaded by HomeScreen on web only, so it
// never touches native. Draws into a <canvas> absolutely positioned behind
// the content inside the #bg-canvas-host element.
export default function startBg() {
  if (typeof document === 'undefined') return;
  const host = document.querySelector('[data-testid="bg-canvas-host"], [nativeID="bg-canvas-host"], #bg-canvas-host')
    || document.querySelector('div[nativeID="bg-canvas-host"]');
  const parent = host || document.body;
  if (document.getElementById('agri-bg-canvas')) return; // already running

  const canvas = document.createElement('canvas');
  canvas.id = 'agri-bg-canvas';
  Object.assign(canvas.style, {
    position: 'absolute', inset: '0', width: '100%', height: '100%',
    zIndex: '0', pointerEvents: 'none', opacity: '0.5',
  });
  parent.style.position = parent.style.position || 'relative';
  parent.insertBefore(canvas, parent.firstChild);
  const ctx = canvas.getContext('2d');

  let w, h, pts;
  const N = 42;
  function resize() {
    w = canvas.width = parent.clientWidth;
    h = canvas.height = parent.clientHeight;
  }
  function init() {
    pts = Array.from({ length: N }, () => ({
      x: Math.random() * w, y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.25, vy: (Math.random() - 0.5) * 0.25,
    }));
  }
  function step() {
    ctx.clearRect(0, 0, w, h);
    for (const p of pts) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
    }
    ctx.fillStyle = 'rgba(160,175,190,0.7)';
    for (const p of pts) { ctx.beginPath(); ctx.arc(p.x, p.y, 1.4, 0, 7); ctx.fill(); }
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      const a = pts[i], b = pts[j];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (d < 130) {
        ctx.strokeStyle = `rgba(120,135,150,${(1 - d / 130) * 0.25})`;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    }
    requestAnimationFrame(step);
  }
  resize(); init(); step();
  window.addEventListener('resize', () => { resize(); init(); });
}
