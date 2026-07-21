import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';

export default function FeatureCard({ icon, tag, title, desc, model, accuracy, inputs = [], to }) {
  const [hovered, setHovered] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="glass-card glass-card-scanner p-6 flex flex-col h-full cursor-pointer"
    >
      <div className="flex items-start gap-3.5 mb-5">
        <div className="w-14 h-14 rounded-xl bg-chrome/8 border border-chrome/15 flex items-center justify-center text-3xl shrink-0">
          {icon}
        </div>
        <span className="font-mono text-[11px] text-chrome tracking-widest uppercase mt-1.5">
          {tag}
        </span>
      </div>

      <h3 className="font-heading text-2xl font-bold text-white-soft mb-2">{title}</h3>
      <p className="font-body text-mid text-[13.5px] leading-relaxed flex-grow">{desc}</p>

      {/* Model pill */}
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <span className="bg-amber/15 text-amber-light font-mono text-[11px] px-3 py-1 rounded-full">
          {model}
        </span>
        <span className="text-leaf font-mono text-xs">
          <span className="inline-block w-5 h-1.5 rounded-full bg-leaf mr-1.5 align-middle" />
          {accuracy}
        </span>
      </div>

      {/* Expandable inputs */}
      <AnimatePresence>
        {hovered && inputs.length > 0 && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="mt-4 pt-3 border-t border-chrome/10 space-y-1">
              {inputs.map((inp, i) => (
                <div key={i} className="text-mid text-xs font-body flex items-start gap-2">
                  <span className="text-chrome mt-0.5">›</span>
                  <span>{inp}</span>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <Link
        to={to}
        className="mt-5 inline-flex items-center gap-2 font-heading font-semibold text-sm text-chrome hover:text-white-soft transition-colors group"
      >
        Try it
        <span className="group-hover:translate-x-1 transition-transform">→</span>
      </Link>
    </motion.div>
  );
}
