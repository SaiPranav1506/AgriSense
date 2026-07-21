import { useRef, useEffect } from 'react';

export default function ParticleField() {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let raf;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const nodes = Array.from({ length: 60 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      radius: Math.random() * 2.2 + 0.8,
      alpha: Math.random() * 0.4 + 0.15,
      pulsePhase: Math.random() * Math.PI * 2,
    }));

    let mouseX = -1000;
    window.addEventListener('mousemove', e => (mouseX = e.clientX));

    const tick = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // faint grid
      ctx.strokeStyle = 'rgba(185,195,207,0.035)';
      ctx.lineWidth = 1;
      const spacing = 64;
      for (let x = 0; x < canvas.width; x += spacing) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += spacing) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }

      // connections
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 110) {
            const a = (1 - dist / 110) * 0.18;
            ctx.strokeStyle = `rgba(185,195,207,${a})`;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.stroke();
          }
        }
      }

      // nodes
      for (const n of nodes) {
        const pulse = Math.sin(n.pulsePhase) * 0.5 + 0.5;
        const r = n.radius * (1 + pulse * 0.3);

        // mouse repulsion
        const mdx = n.x - mouseX;
        const mdy = n.y - (canvas.height / 2);
        const md = Math.sqrt(mdx * mdx + mdy * mdy);
        if (md < 130 && md > 0) {
          const force = (130 - md) / 130 * 0.04;
          n.vx += (mdx / md) * force;
          n.vy += (mdy / md) * force;
        }

        n.vx *= 0.97;
        n.vy *= 0.97;
        n.x += n.vx;
        n.y += n.vy;
        n.pulsePhase += 0.015;

        if (n.x < 0) n.x = canvas.width;
        if (n.x > canvas.width) n.x = 0;
        if (n.y < 0) n.y = canvas.height;
        if (n.y > canvas.height) n.y = 0;

        // glow
        const grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r * 4);
        grad.addColorStop(0, `rgba(185,195,207,${n.alpha * (0.7 + pulse * 0.3)})`);
        grad.addColorStop(1, 'rgba(185,195,207,0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(n.x, n.y, r * 4, 0, Math.PI * 2);
        ctx.fill();
      }

      raf = requestAnimationFrame(tick);
    };

    tick();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', () => {});
    };
  }, []);

  return <canvas ref={ref} className="particle-canvas" />;
}
