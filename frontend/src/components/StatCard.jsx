import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';

export default function StatCard({ value, label, unit = '', delay = 0 }) {
  const isNum = /^[0-9.,+]+$/.test(String(value).replace(/[^0-9.,+]/g, ''));
  const [display, setDisplay] = useState('0');
  const started = useRef(false);

  useEffect(() => {
    if (!isNum) { setDisplay(value); return; }
    const target = parseFloat(String(value).replace(/[^0-9.]/g, ''));
    const duration = 1500;
    const start = performance.now();
    const numDigits = (target.toString().match(/\./) ?? [])[0] ? 2 : 0;

    const tick = (now) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay((target * eased).toFixed(numDigits));
      if (progress < 1) requestAnimationFrame(tick);
    };
    const raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, isNum]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay }}
      className="glass-card p-5 text-center"
    >
      <div className="font-heading text-chrome text-3xl font-bold mb-1">
        {String(display).startsWith('+') || String(display).startsWith('~') ? '' : ''}
        {display}
        {unit}
      </div>
      <div className="text-mid font-mono text-[11px] tracking-wider uppercase">{label}</div>
    </motion.div>
  );
}
