import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, Leaf } from 'lucide-react';

const LINKS = [
  { label: 'Home', to: '/' },
  { label: 'Features', to: '/#features' },
  { label: 'About', to: '/#about' },
  { label: 'Live Demo', to: '/#get-started' },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => { setOpen(false); }, [location.pathname]);

  return (
    <motion.nav
      initial={{ y: -80 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'bg-surface/90 backdrop-blur-2xl shadow-lg shadow-black/40 border-b border-silver/10'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2.5 group">
          <span className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-silver-mid opacity-50" />
            <span className="relative inline-flex rounded-full h-3 w-3 bg-silver" />
          </span>
          <span className="font-heading font-bold text-xl text-white-soft tracking-tight">
            AgriSense
          </span>
        </Link>

        <div className="hidden md:flex items-center gap-8">
          {LINKS.map(l => (
            <Link key={l.to} to={l.to}
              className="text-sm font-medium text-mid hover:text-silver-mid transition-colors duration-200">
              {l.label}
            </Link>
          ))}
          <span className="flex items-center gap-2 text-silver-mid font-mono text-[11px] tracking-wider">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-silver opacity-50" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-silver" />
            </span>
            CLIMATE FEED ACTIVE
          </span>
        </div>

        <button onClick={() => setOpen(!open)} className="md:hidden text-white-soft p-2" aria-label="Toggle menu">
          {open ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ x: '100%', opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: '100%', opacity: 0 }}
            transition={{ duration: 0.35, ease: 'easeInOut' }}
            className="md:hidden fixed inset-0 top-[68px] bg-surface/95 backdrop-blur-2xl z-40 flex flex-col items-center gap-8 pt-16"
          >
            {LINKS.map(l => (
              <Link key={l.to} to={l.to}
                className="font-heading text-2xl text-white-soft hover:text-silver-mid transition-colors">
                {l.label}
              </Link>
            ))}
            <span className="text-silver-mid font-mono text-sm tracking-wider">● CLIMATE FEED ACTIVE</span>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  );
}
