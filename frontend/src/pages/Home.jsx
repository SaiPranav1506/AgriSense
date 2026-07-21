import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ChevronDown, Leaf } from 'lucide-react';
import SectionWrapper from '../components/SectionWrapper';
import StatCard from '../components/StatCard';
import FeatureCard from '../components/FeatureCard';
import LiveTicker from '../components/LiveTicker';
import SectionDivider from '../components/SectionDivider';

const HERO_VAR = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.12 },
  },
};

const UP = {
  hidden: { opacity: 0, y: 24 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.65, ease: [0.22, 1, 0.36, 1] },
  },
};

function LeafIcon() {
  return (
    <svg
      width="80"
      height="80"
      viewBox="0 0 80 80"
      fill="none"
      className="mx-auto block"
      aria-hidden="true"
    >
      <circle cx="40" cy="40" r="37" stroke="#9ca3af" strokeWidth="1.5" opacity="0.35" />
      <circle cx="40" cy="40" r="28" fill="#6b7280" opacity="0.15" />
      <circle cx="40" cy="40" r="20" fill="url(#lg1)" opacity="0.5" />
      <defs>
        <linearGradient id="lg1" x1="20" y1="20" x2="60" y2="60">
          <stop offset="0%" stopColor="#9ca3af" />
          <stop offset="100%" stopColor="#9ca3af" />
        </linearGradient>
      </defs>
      <path
        d="M40 16 C40 16 54 28 54 40 C54 48.7 47.1 55.6 40 55.6 C32.9 55.6 26 48.7 26 40 C26 28 40 16 40 16Z"
        fill="#9ca3af"
        opacity="0.85"
      />
      <path
        d="M40 20.5 C40 20.5 51.5 29.8 51.5 40 C51.5 47.9 46.1 54 40 54 C33.9 54 28.5 47.9 28.5 40 C28.5 29.8 40 20.5 40 20.5Z"
        stroke="#f1f5f9"
        strokeWidth="1"
        fill="none"
        opacity="0.5"
      />
      <line x1="40" y1="20.5" x2="40" y2="12" stroke="#9ca3af" strokeWidth="2" strokeLinecap="round" opacity="0.9" />
      <circle cx="40" cy="11" r="2.5" fill="#9ca3af" />
      <line x1="34" y1="28" x2="27" y2="21" stroke="#9ca3af" strokeWidth="1.5" strokeLinecap="round" opacity="0.65" />
      <line x1="46" y1="28" x2="53" y2="21" stroke="#9ca3af" strokeWidth="1.5" strokeLinecap="round" opacity="0.65" />
      <line x1="31" y1="40" x2="23" y2="40" stroke="#9ca3af" strokeWidth="1.3" strokeLinecap="round" opacity="0.55" />
      <line x1="49" y1="40" x2="57" y2="40" stroke="#9ca3af" strokeWidth="1.3" strokeLinecap="round" opacity="0.55" />
      <line x1="34" y1="50" x2="28" y2="56" stroke="#9ca3af" strokeWidth="1.3" strokeLinecap="round" opacity="0.6" />
      <line x1="46" y1="50" x2="52" y2="56" stroke="#9ca3af" strokeWidth="1.3" strokeLinecap="round" opacity="0.6" />
    </svg>
  );
}

const CHALLENGES = [
  {
    icon: '🌾',
    title: 'No real-time crop advisory',
    text: "Farmers decide based on last season's prices or family tradition — not live soil chemistry or climate trajectories. A wrong choice wastes an entire season.",
    tag: 'Module A — XGBoost',
  },
  {
    icon: '📉',
    title: 'Unpredictable yield estimates',
    text: 'Without accurate forecasts, farmers cannot plan storage, negotiate prices, or access credit. Most estimates are guesswork — not climate-driven modelling.',
    tag: 'Module B — CNN-LSTM',
  },
  {
    icon: '🔬',
    title: 'Late disease detection',
    text: 'Manual inspection catches outbreaks 5–7 days after onset, by which time up to 30% of a crop can already be compromised.',
    tag: 'Module C — MobileNetV2',
  },
  {
    icon: '📅',
    title: 'Static calendar-based farming',
    text: "Planting and irrigation still follow fixed almanac dates — built for stable climates that no longer exist.",
    tag: 'Modules A + B',
  },
  {
    icon: '🧩',
    title: 'No integrated advisory platform',
    text: 'Existing tools address one task at a time. AgriSense unifies crop selection, yield projection, and disease detection in one interface.',
    tag: 'All Three Modules',
  },
];

export default function Home() {
  return (
    <div className="relative">
      {/* ── HERO ── */}
      <section
        id="hero"
        className="relative flex flex-col items-center justify-center overflow-hidden"
        style={{ minHeight: '100vh' }}
      >
        <div className="absolute inset-0 bg-gradient-to-b from-chrome/10 via-surface to-deep" />

        <motion.div
          variants={HERO_VAR}
          initial="hidden"
          animate="visible"
          className="relative z-10 text-center max-w-4xl mx-auto px-6 py-32"
        >
          <motion.div variants={UP} className="mb-8">
            <LeafIcon />
          </motion.div>

          <motion.h1
            variants={UP}
            className="font-heading text-5xl md:text-6xl lg:text-[64px] font-bold leading-[1.08] tracking-tight text-white-soft mb-3"
          >
            Agri<span className="text-gradient">Sense</span>
          </motion.h1>

          <motion.p
            variants={UP}
            className="font-mono text-[11px] text-chrome tracking-[0.25em] uppercase mb-4"
          >
            Smart Climate Agriculture · AI System
          </motion.p>

          <motion.p
            variants={UP}
            className="font-body text-base md:text-lg text-mid max-w-[560px] mx-auto mb-9 leading-relaxed"
          >
            AI-powered crop advisory built on real climate data — precise yields,
            early disease alerts, smart crop selection, and climate-driven
            decisions for every field across India.
          </motion.p>

          <motion.div variants={UP} className="flex flex-wrap justify-center gap-4 mb-14">
            <Link
              to="/#features"
              className="leaf-btn px-7 py-3.5 font-heading text-base"
            >
              Explore Features ↓
            </Link>
            <Link
              to="/#get-started"
              className="ghost-btn px-7 py-3.5 font-heading text-base"
            >
              Try Live Demo →
            </Link>
          </motion.div>

          <motion.div variants={UP} className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-14">
            <StatCard value="140M+" label="Farmers in India" delay={0} />
            <StatCard value="58%" label="Workforce in Agriculture" delay={0.1} />
            <StatCard value="₹50K Cr+" label="Annual Crop Loss" delay={0.2} />
            <StatCard value="3" label="AI Models Unified" delay={0.3} />
          </motion.div>

          <motion.div variants={UP} className="flex flex-col items-center gap-1">
            <span className="text-muted text-xs font-body">Scroll to explore</span>
            <ChevronDown className="text-chrome animate-bounce mt-0.5" size={18} />
          </motion.div>
        </motion.div>
      </section>

      {/* ── ABOUT ── */}
      <SectionWrapper id="about" className="bg-surface py-24">
        <div className="max-w-7xl mx-auto px-6 grid md:grid-cols-5 gap-14 items-start">
          <div className="md:col-span-3">
            <p className="font-mono text-[11px] text-chrome tracking-[0.2em] uppercase mb-3">
              About Indian Agriculture
            </p>
            <h2 className="font-heading text-4xl md:text-[40px] font-bold text-white-soft leading-tight mb-8">
              The backbone of a billion lives — under{' '}
              <span className="text-gradient">unprecedented</span> stress
            </h2>

            <div className="space-y-5 font-body text-mid leading-relaxed text-[15px]">
              <p>
                Agriculture is the foundation of India&apos;s economy, supporting
                over 140 million farming households and contributing roughly 18% to
                the national GDP. With more than half the country&apos;s workforce
                employed in farm-related activities, the sector feeds not just
                India&apos;s 1.4 billion population but also drives significant
                export revenue across rice, wheat, sugarcane, cotton, and
                horticultural produce.
              </p>
              <p>
                Yet Indian agriculture operates under extraordinary climate risk.
                The country depends on a monsoon season that delivers nearly 70% of
                annual rainfall in just four months. Any deviation — a delayed
                onset, uneven distribution, or abrupt withdrawal — cascades into
                drought in some regions and flooding in others, often within the
                same season. Rising global temperatures are making these extremes
                more frequent and less predictable year on year.
              </p>
              <p>
                The economic consequences are severe. India loses an estimated
                ₹50,000 crore annually to crop failures, pest outbreaks, and soil
                degradation. Smallholder farmers — those cultivating less than two
                hectares — bear the heaviest burden, lacking the insurance, credit,
                and advisory infrastructure that larger agricultural enterprises
                enjoy. A single failed harvest can push a farming family into debt
                that takes years to recover from.
              </p>
              <p>
                Despite the scale of the problem, precision agricultural technology
                remains accessible to fewer than 2% of smallholder farmers in
                India. Most rely on knowledge passed down through generations —
                effective in stable climates, but increasingly unreliable as
                weather patterns shift. The gap between what AI-driven advisory
                systems can deliver and what farmers actually receive represents
                one of the most consequential digital divides in the country.
              </p>
            </div>
          </div>

          <div className="md:col-span-2 flex flex-col gap-4 mt-2">
            {[
              { val: '18%', label: 'contribution to GDP' },
              { val: '70%', label: 'rainfall in just 4 months' },
              { val: '<2%', label: 'farmers with precision tools' },
              { val: '4 mos', label: 'window that decides the year' },
            ].map((s, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1, duration: 0.5 }}
                className="glass-card p-5"
              >
                <div className="font-heading text-[42px] font-bold text-gradient">
                  {s.val}
                </div>
                <div className="text-mid font-body text-sm">{s.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </SectionWrapper>

      <SectionDivider />

      {/* ── CHALLENGES ── */}
      <SectionWrapper id="challenges" className="bg-elevated/40 py-24">
        <div className="max-w-7xl mx-auto px-6">
          <p className="font-mono text-[11px] text-chrome tracking-[0.2em] uppercase mb-3">
            The Real Problem
          </p>
          <h2 className="font-heading text-4xl font-bold text-white-soft mb-4">
            Five challenges that AI can solve right now
          </h2>
          <p className="text-mid font-body mb-12 max-w-2xl">
            Each of these has a direct mapping to one of our three AI modules.
          </p>

          <div className="grid md:grid-cols-2 gap-6">
            {CHALLENGES.map((c, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.08, duration: 0.5 }}
                className="glass-card glass-card-scanner p-6"
              >
                <span className="text-3xl mb-3 block">{c.icon}</span>
                <h3 className="font-heading text-xl font-semibold text-white-soft mb-2">
                  {c.title}
                </h3>
                <p className="text-mid font-body text-sm leading-relaxed mb-4">
                  {c.text}
                </p>
                <span className="font-mono text-[11px] text-chrome bg-chrome/10 px-2.5 py-1 rounded-full">
                  {c.tag}
                </span>
              </motion.div>
            ))}
          </div>
        </div>
      </SectionWrapper>

      {/* ── TICKER ── */}
      <LiveTicker />

      {/* ── FEATURES ── */}
      <SectionWrapper id="features" className="bg-surface py-24">
        <div className="max-w-7xl mx-auto px-6">
          <p
            className="font-mono text-[11px] text-chrome tracking-[0.2em] uppercase mb-3 text-center"
          >
            AI Intelligence Modules
          </p>
          <h2 className="font-heading text-4xl font-bold text-white-soft text-center mb-3">
            Three models. One system. Every decision covered.
          </h2>
          <p className="text-mid font-body text-center mb-14 max-w-2xl mx-auto">
            Hover each card to see the model scanning live. Click <em>Try it</em> to
            open that feature.
          </p>

          <div className="grid md:grid-cols-3 gap-7">
            <FeatureCard
              icon="🌱"
              tag="Module A"
              title="Crop Recommendation"
              desc="Ingests soil NPK, pH, live temperature, humidity, and rainfall to rank 22 crops by suitability. Runs in under 5ms at inference."
              model="XGBoost Classifier"
              accuracy="95% classification accuracy"
              inputs={[
                'Nitrogen, Phosphorus, Potassium',
                'Soil pH (3.5–9.5)',
                'Temperature, Humidity, Rainfall',
                'Live climate via Open-Meteo API',
              ]}
              to="/crop"
            />
            <FeatureCard
              icon="📈"
              tag="Module B"
              title="Yield Forecast"
              desc="A 30-day rolling climate window feeds CNN layers for pattern extraction, LSTM for seasonal memory, and Attention for monsoon peak weighting."
              model="CNN-LSTM + Attention"
              accuracy="R² = 0.93"
              inputs={[
                '30-day temperature sequence',
                '30-day rainfall sequence',
                'Crop type, area, season',
                'Real-time API window refresh',
              ]}
              to="/yield"
            />
            <FeatureCard
              icon="🔬"
              tag="Module C"
              title="Disease Detection"
              desc="Upload a leaf photo. MobileNetV2 classifies it against 38 classes across 14 crop species in under 100ms on CPU."
              model="MobileNetV2 Fine-tuned"
              accuracy="96% on 54K images"
              inputs={[
                'Leaf photo upload (JPG/PNG)',
                'Resized to 224×224 internally',
                'Returns class + confidence score',
                '14 MB — runs CPU-only on Flask',
              ]}
              to="/disease"
            />
          </div>

          {/* Accuracy bars */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mt-14 glass-card p-7 space-y-5"
          >
            {[
              { name: 'XGBoost — Crop Recommendation', pct: 95 },
              { name: 'CNN-LSTM — Yield Forecast', pct: 93 },
              { name: 'MobileNetV2 — Disease Detection', pct: 96 },
            ].map((row, i) => (
              <div key={i}>
                <div className="flex justify-between font-mono text-xs text-mid mb-1.5">
                  <span>{row.name}</span>
                  <span className="text-gradient font-bold">{row.pct}%</span>
                </div>
                <div className="h-2.5 rounded-full bg-white/5 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    whileInView={{ width: `${row.pct}%` }}
                    viewport={{ once: true }}
                    transition={{
                      duration: 1.4,
                      delay: i * 0.15,
                      ease: [0.22, 1, 0.36, 1],
                    }}
                    className="h-full rounded-full bg-gradient-to-r from-chrome to-silver-mid"
                  />
                </div>
              </div>
            ))}
          </motion.div>
        </div>
      </SectionWrapper>

      <SectionDivider />

      {/* ── CTA BLOCK ── */}
      <SectionWrapper id="get-started" className="bg-deep py-24">
        <div
          className="max-w-4xl mx-auto px-6 text-center relative overflow-hidden rounded-3xl py-20 px-8"
          style={{
            background:
              'radial-gradient(ellipse at center, rgba(99,102,241,0.10) 0%, rgba(6,8,15,0) 70%)',
          }}
        >
          <h2 className="font-heading text-4xl font-bold text-white-soft mb-4">
            Ready to make data-driven farming decisions?
          </h2>
          <p className="text-mid font-body mb-12">
            Pick any module below to start a live prediction.
          </p>

          <div className="grid sm:grid-cols-3 gap-5">
            {[
              { icon: '🌱', label: 'Crop Recommendation', desc: 'Find your ideal crop', to: '/crop' },
              { icon: '📈', label: 'Yield Forecast', desc: 'Predict your harvest', to: '/yield' },
              { icon: '🔬', label: 'Disease Detection', desc: 'Diagnose plant health', to: '/disease' },
            ].map((m, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
              >
                <Link
                  to={m.to}
                  className="glass-card p-6 flex flex-col items-center gap-3 block"
                >
                  <span className="text-4xl">{m.icon}</span>
                  <span className="font-heading font-semibold text-white-soft">
                    {m.label}
                  </span>
                  <span className="text-mid text-sm">{m.desc}</span>
                  <span className="text-gradient font-heading text-sm mt-2">
                    Launch →
                  </span>
                </Link>
              </motion.div>
            ))}
          </div>

          <footer className="mt-20 pt-8 border-t border-chrome/10 text-muted font-body text-xs">
            AgriSense · Built with TensorFlow, XGBoost, and Flask
          </footer>
        </div>
      </SectionWrapper>
    </div>
  );
}
