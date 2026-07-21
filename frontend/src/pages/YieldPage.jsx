import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Loader2, MapPin } from 'lucide-react';
import { API_BASE } from '../config';

const SEASONS = ['Kharif 23', 'Rabi 23', 'Kharif 24'];
const SEASON_BARS = [3.2, 4.1, 3.8];

const PHASES = [
  { label: 'Sowing', pct: 15 },
  { label: 'Vegetative', pct: 40 },
  { label: 'Reproductive', pct: 75 },
  { label: 'Maturity', pct: 100 },
];

export default function YieldPage() {
  const [areas, setAreas] = useState([]);
  const [items, setItems] = useState([]);
  const [form, setForm] = useState({
    area_code: 57, item_code: 7, area_ha: 2.5,
    rainfall: 1200, avg_temp: 27, year: 2024, pesticides: 100,
  });
  const [useWeather, setUseWeather] = useState(false);
  const [lat, setLat] = useState(17.36);
  const [lon, setLon] = useState(78.48);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/encoders`)
      .then(r => r.json())
      .then(d => {
        setAreas(d.areas);
        setItems(d.items);
        setForm(f => ({
          ...f,
          area_code: d.area_map?.['India'] ?? 0,
          item_code: d.item_map?.['Rice, paddy'] ?? 0,
        }));
      })
      .catch(() => {});
  }, []);

  const ch = (k, v) => setForm(f => ({ ...f, [k]: parseFloat(v) }));

  const submit = async () => {
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const body = { ...form };
      if (useWeather) { body.lat = parseFloat(lat); body.lon = parseFloat(lon); }
      const r = await fetch(`${API_BASE}/predict-yield-raw`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error((await r.json()).error || r.statusText);
      setResult(await r.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.4, ease: 'easeInOut' }}
      className="min-h-screen pt-28 pb-16 px-6"
    >
      <div className="max-w-6xl mx-auto">
        {/* Form card */}
        <div className="glass-card p-8 mb-10">
          <h2 className="font-heading text-2xl font-bold text-white-soft mb-2">
            Forecast Yield
          </h2>
          <p className="text-mid font-body text-sm mb-6">
            Predict crop yield in tonnes per hectare based on location and conditions.
          </p>

          <div className="grid sm:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="font-mono text-[11px] text-mid uppercase block mb-1">
                Country / Area
              </label>
              <select
                value={form.area_code}
                onChange={e => ch('area_code', e.target.value)}
                className="w-full"
              >
                {areas.map((a, i) => (
                  <option key={a} value={i}>{a}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="font-mono text-[11px] text-mid uppercase block mb-1">
                Crop
              </label>
              <select
                value={form.item_code}
                onChange={e => ch('item_code', e.target.value)}
                className="w-full"
              >
                {items.map((it, i) => (
                  <option key={it} value={i}>{it}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="font-mono text-[11px] text-mid uppercase block mb-1">
                Area (ha)
              </label>
              <input type="number" value={form.area_ha} onChange={e => ch('area_ha', e.target.value)} step="0.1" />
            </div>
            <div>
              <label className="font-mono text-[11px] text-mid uppercase block mb-1">
                Avg Rainfall (mm)
              </label>
              <input type="number" value={form.rainfall} onChange={e => ch('rainfall', e.target.value)} />
            </div>
            <div>
              <label className="font-mono text-[11px] text-mid uppercase block mb-1">
                Avg Temp (°C)
              </label>
              <input type="number" value={form.avg_temp} onChange={e => ch('avg_temp', e.target.value)} step="0.1" />
            </div>
            <div>
              <label className="font-mono text-[11px] text-mid uppercase block mb-1">
                Year
              </label>
              <input type="number" value={form.year} onChange={e => ch('year', e.target.value)} step={1} />
            </div>
          </div>

          <label className="flex items-center gap-2 text-sm text-mid cursor-pointer">
            <input
              type="checkbox"
              checked={useWeather}
              onChange={e => setUseWeather(e.target.checked)}
              className="accent-chrome"
            />
            <MapPin size={15} />
            Use live 30-day climate window
          </label>

          <AnimatePresence>
            {useWeather && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="grid grid-cols-2 gap-3 mt-3 overflow-hidden"
              >
                <div>
                  <label className="font-mono text-[11px] text-mid">Latitude</label>
                  <input type="number" value={lat} onChange={e => setLat(parseFloat(e.target.value))} />
                </div>
                <div>
                  <label className="font-mono text-[11px] text-mid">Longitude</label>
                  <input type="number" value={lon} onChange={e => setLon(parseFloat(e.target.value))} />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <button
            onClick={submit}
            disabled={loading}
            className="leaf-btn w-full py-3.5 mt-5 font-heading text-base flex items-center justify-center gap-2"
          >
            {loading ? <><Loader2 className="animate-spin" size={18} /> Forecasting...</> : 'Forecast Yield'}
          </button>
          {!loading && <div className="h-3.5" />}
        </div>

        {/* Results */}
        <AnimatePresence mode="wait">
          {result && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
              <div className="grid sm:grid-cols-3 gap-5">
                {[
                  { label: 'Yield / Hectare', val: `${result.yield_per_ha?.toFixed(3) || '—'} t/ha`, accent: 'text-amber' },
                  { label: 'Total Production', val: `${result.total_yield?.toFixed(3) || '—'} t`, accent: 'text-white-soft' },
                  { label: 'Unit', val: result.unit || 't/ha', accent: 'text-mid' },
                ].map((s, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.1 }}
                    className="glass-card p-5 text-center"
                  >
                    <p className="text-mid font-mono text-[11px] tracking-wider uppercase">{s.label}</p>
                    <p className={`font-mono text-2xl font-bold mt-2 ${s.accent}`}>{s.val}</p>
                  </motion.div>
                ))}
              </div>

              <div className="glass-card p-7">
                <p className="font-mono text-[11px] text-chrome tracking-widest uppercase mb-5">
                  Seasonal Comparison
                </p>
                <div className="space-y-4">
                  {SEASONS.map((s, i) => (
                    <div key={i} className="flex items-center gap-4">
                      <span className="font-mono text-xs text-mid w-24 shrink-0">{s}</span>
                      <div className="flex-1 h-10 bg-white/5 rounded-lg relative overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${(SEASON_BARS[i] / 5) * 100}%` }}
                          transition={{ duration: 1, delay: i * 0.15 }}
                          className={`h-full rounded-lg ${i === SEASONS.length - 1 ? 'bg-gradient-to-r from-chrome to-silver-mid' : 'bg-chrome/40'}`}
                        />
                        <span className="absolute right-3 top-1/2 -translate-y-1/2 font-mono text-xs text-white-soft font-bold">
                          {SEASON_BARS[i]} t/ha
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="glass-card p-7">
                <p className="font-mono text-[11px] text-chrome tracking-widest uppercase mb-5">
                  Crop Development Phases
                </p>
                <div className="space-y-4">
                  {PHASES.map((p, i) => (
                    <div key={i} className="flex items-center gap-4">
                      <span className="font-mono text-xs text-mid w-28 shrink-0">{p.label}</span>
                      <div className="flex-1 h-3 rounded-full bg-white/5 overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${p.pct}%` }}
                          transition={{ duration: 1.2, delay: i * 0.15 }}
                          className="h-full rounded-full bg-gradient-to-r from-chrome to-silver-mid"
                        />
                      </div>
                      <span className="font-mono text-xs text-chrome font-bold w-12 text-right">{p.pct}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {!result && !error && (
            <div className="glass-card p-12 text-center">
              <p className="text-muted italic font-body">Your yield forecast will appear here</p>
            </div>
          )}

          {error && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-card border-amber/40 p-6">
              <p className="font-heading font-semibold text-amber mb-2">API Error</p>
              <p className="text-mid text-sm mb-2">{error}</p>
              <p className="text-xs text-muted">Is Flask running?</p>
            </motion.div>
          )}
        </AnimatePresence>

        <Link to="/" className="ghost-btn inline-flex items-center gap-2 px-5 py-2.5 mt-8 text-sm font-heading">
          {'<-'} Back to Home
        </Link>
      </div>
    </motion.div>
  );
}
