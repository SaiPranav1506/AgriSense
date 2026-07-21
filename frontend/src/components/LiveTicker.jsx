import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

const TICKER_DATA = [
  { label: 'TEMP AIR', value: '27.4 °C' },
  { label: 'HUMIDITY', value: '68 %' },
  { label: 'RAIN', value: '112 mm' },
  { label: 'TOP CROP', value: 'Rice' },
  { label: 'PRED YIELD', value: '3.8 t/ha' },
  { label: 'ALERT', value: 'Blast HIGH' },
  { label: 'SOIL N', value: '82 kg/ha' },
  { label: 'pH LEVEL', value: '6.5' },
  { label: 'WIND', value: '14 km/h' },
  { label: 'API', value: 'Open-Meteo LIVE' },
];

function jitter(val) {
  const n = parseFloat(val);
  if (isNaN(n)) return val;
  const nudged = (n + (Math.random() - 0.5) * n * 0.05).toFixed(n < 10 ? 1 : 0);
  return String(nudged);
}

export default function LiveTicker() {
  const [data, setData] = useState(TICKER_DATA);

  useEffect(() => {
    const id = setInterval(() => {
      setData(prev => prev.map(d => ({
        ...d,
        value: d.label === 'ALERT' ? d.value : jitter(d.value),
      })));
    }, 3000);
    return () => clearInterval(id);
  }, []);

  const doubled = [...data, ...data];

  return (
    <div className="w-full overflow-hidden bg-silver/[0.04] border-y border-silver/10 py-2.5">
      <motion.div
        className="flex gap-10 w-max"
        animate={{ x: ['0%', '-50%'] }}
        transition={{ duration: 40, repeat: Infinity, ease: 'linear' }}
      >
        {doubled.map((item, i) => (
          <span key={i} className="flex items-center gap-3 text-sm whitespace-nowrap font-mono">
            <span className="text-muted text-[11px] tracking-wider uppercase">{item.label}</span>
            <span className="text-silver-mid font-semibold">{item.value}</span>
          </span>
        ))}
      </motion.div>
    </div>
  );
}
