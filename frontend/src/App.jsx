import { Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import Navbar from './components/Navbar';
import ParticleField from './components/ParticleField';
import ErrorBoundary from './components/ErrorBoundary';
import Home from './pages/Home';
import CropPage from './pages/CropPage';
import YieldPage from './pages/YieldPage';
import DiseasePage from './pages/DiseasePage';
import NotFound from './pages/NotFound';

const pageTransition = {
  initial: { opacity: 0, y: 18 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -14 },
  transition: { duration: 0.38, ease: 'easeInOut' },
};

function AnimatedBg() {
  return (
    <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
      <div className="bg-mesh">
        <div className="bg-mesh-layer" />
        <div className="bg-mesh-layer" />
        <div className="bg-mesh-layer" />
        <div className="bg-mesh-grid" />
      </div>
      <div className="bg-orb bg-orb--1" />
      <div className="bg-orb bg-orb--2" />
      <div className="bg-orb bg-orb--3" />
    </div>
  );
}

function AnimatedRoutes() {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<ErrorBoundary><motion.div {...pageTransition}><Home /></motion.div></ErrorBoundary>} />
        <Route path="/crop" element={<ErrorBoundary><motion.div {...pageTransition}><CropPage /></motion.div></ErrorBoundary>} />
        <Route path="/yield" element={<ErrorBoundary><motion.div {...pageTransition}><YieldPage /></motion.div></ErrorBoundary>} />
        <Route path="/disease" element={<ErrorBoundary><motion.div {...pageTransition}><DiseasePage /></motion.div></ErrorBoundary>} />
        <Route path="*" element={<motion.div {...pageTransition}><NotFound /></motion.div>} />
      </Routes>
    </AnimatePresence>
  );
}

export default function App() {
  return (
    <>
      <AnimatedBg />
      <ParticleField />
      <Navbar />
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: 'rgba(12, 14, 22, 0.82)',
            backdropFilter: 'blur(18px)',
            border: '1px solid rgba(185, 195, 207, 0.12)',
            color: '#f3f4f6',
            borderRadius: '12px',
            fontFamily: "'Inter', sans-serif",
          },
          success: { iconTheme: { primary: '#7dd3a8', secondary: '#030406' } },
          error: { iconTheme: { primary: '#e2b96f', secondary: '#030406' } },
        }}
      />
      <AnimatedRoutes />
    </>
  );
}