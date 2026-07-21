import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Leaf, MapPin, Loader2 } from 'lucide-react';
import { API_BASE } from '../config';

const CROP_TIPS = {
  rice: 'Ensure flooded paddy conditions. Apply split urea doses at tillering stage.',
  maize: 'Ensure well-drained soil. Apply nitrogen in two splits for best yield.',
  chickpea: 'Sow in rabi season. Ensure well-drained soil; avoid waterlogging.',
  kidneybeans: 'Needs moderate rainfall. Supports nitrogen fixation - rotate annually.',
  pigeonpeas: 'Deep-rooted; suitable for marginal soils. Drought tolerant.',
  mothbeans: 'Drought-hardy legume. Ideal for arid zones with poor soils.',
  mungbean: 'Short-duration crop. Can be grown in rotation with rice-wheat systems.',
  blackgram: 'Requires moderate moisture and well-drained loam soil.',
  lentil: 'Cool-season pulse. Grows best on sandy loam with neutral pH.',
  pomegranate: 'Drought-tolerant once established. Fruit quality peaks with <20% RH.',
  banana: 'Needs warm humid conditions. Mulch heavily and protect from strong wind.',
  mango: 'Tropical climate required. Water young trees; suppress watering after fruiting begins.',
  grapes: 'Requires well-drained soil and controlled irrigation. Prune heavily annually.',
  watermelon: 'Warm-season crop. Light, sandy loam ideal. Reduce water near harvest.',
  muskmelon: 'Requires warm, dry climate. Grow on raised beds for good drainage.',
  apple: 'Requires 1,000+ hours below 7C (chilling). Plant on slopes for drainage.',
  orange: 'Subtropical - needs mild winters and dry summers. Irrigate weekly.',
  papaya: 'Tropical; annual in temperate. Needs shelter from frost and wind.',
  coconut: 'Coastal tropical. Salt-tolerant. Does not grow inland where temperature falls below 15C.',
  cotton: 'Long hot summer, 200+ frost-free days. Protect from bollworm with IPM.',
  jute: 'Warm humid climate required. Retting needs standing stagnant water.',
  coffee: 'Needs shade. Grows at 600-1500 m in well-drained volcanic or red soil.',
  wheat: 'Sow in rabi season (Oct-Dec). Apply 3 nitrogen splits. Irrigate at crown root initiation.',
  barley: 'Cool-season cereal. Tolerates salinity. Sowing: Oct-Nov. Low water requirement.',
  soybean: 'Sow June-July. Fixes nitrogen - reduce N fertiliser. Watch for yellow mosaic virus.',
  groundnut: 'Sandy loam ideal. Apply gypsum at pegging. Avoid waterlogging at all costs.',
  sugarcane: 'Long-duration (10-12 months). Heavy feeder - needs 200+ kg N/ha. Trash mulch recommended.',
  tea: 'Sloping acidic soils (pH 4.8-6.5). Pluck top 2 leaves + bud. Replant every 30-40 yrs.',
  tobacco: 'Transplant in Nov-Dec. Needs fertile loam. Top 8-10 leaves for Flue-cured variety.',
  sunflower: 'Heat tolerant. Sow Jan-Feb or Jun-Jul. Sowing depth 3-4 cm. Bee pollination boosts yield.',
  mustard: 'Cool-season oilseed. Sow Oct-Nov. Susceptible to aphids - spray dimethoate if needed.',
  turmeric: 'Plant rhizomes in Apr-May. Shade required. Harvest after 9-10 months, boil before drying.',
  ginger: 'Well-drained loam. Rhizome treatment with Carbendazim before planting. Harvest at 8-9 months.',
  chilli: 'Warm-season. Transplant after 4 weeks. Stake plants. Harvest green or red depending on variety.',
  coriander: 'Cool-season spice. Sow Oct-Nov or Feb-Mar. Seed rate 15-20 kg/ha. Stop irrigation at maturity.',
  blackpepper: 'Tropical vine on live supports. Needs >1500mm rain. Harvest green berries for pepper.',
  cardamom: 'Under-tree crop at 600-1500m. Needs 1500+mm rain. Shade and drainage are critical.',
  brinjal: 'Transplant at 4-6 true leaves. Stake tall varieties. Fruit borer - use pheromone traps.',
  tomato: 'Sow in nursery, transplant at 25-30 days. Erect trellis. Spray Borax to prevent blossom end rot.',
  onion: 'Transplant kharif in May, rabi in Nov. Bulb formation triggered by 12-14 h daylight.',
  cabbage: 'Cool-season. Transplant at 30-35 days. Head formation fails above 30C. Watch for diamondback moth.',
  okra: 'Sow Feb-Mar or June-July. Harvest pods at 5-7 days after flowering. Pick every 2 days.',
  rubber: 'Budded stumps planted at 6x3m spacing. Tapping starts at year 5-7. Needs deep well-drained soil.',
  cashew: 'Grows in coastal sandy loam. Plant nuts 8x8m apart. Grafting improves yield. Harvest when apple turns yellow.',
  tapioca: 'Stem cuttings at 1x1m spacing. Harvest at 12 months for starch, 30 months for dry chips.',
  sweetpotato: 'Vine cuttings at 60x15 cm. Red soils preferred. Harvest at 90-110 days. Yields 15-20 t/ha.',
};

export default function CropPage() {
  const [fields, setFields] = useState({
    N: 80, P: 42, K: 43, temperature: 27, humidity: 68, ph: 6.5, rainfall: 120,
  });
  const [useWeather, setUseWeather] = useState(false);
  const [lat, setLat] = useState(17.36);
  const [lon, setLon] = useState(78.48);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const onChange = (k, v) => setFields(f => ({ ...f, [k]: parseFloat(v) }));

  const submit = async () => {
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const body = { ...fields };
      if (useWeather) {
        body.lat = parseFloat(lat);
        body.lon = parseFloat(lon);
      }
      const r = await fetch(`${API_BASE}/predict-crop`, {
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

  const cropKey = result?.recommended_crop?.toLowerCase() || '';
  const tip = CROP_TIPS[cropKey] || 'Follow local agronomy guidelines for best practice.';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.4, ease: 'easeInOut' }}
      className="min-h-screen pt-28 pb-16 px-6"
    >
      <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-10">
        {/* Form */}
        <div className="glass-card p-8">
          <div className="flex items-center gap-3 mb-2">
            <Leaf className="text-chrome" size={26} />
            <h2 className="font-heading text-2xl font-bold text-white-soft">
              Crop Recommendation
            </h2>
          </div>
          <p className="text-mid font-body text-sm mb-7">
            Enter soil and climate parameters to find the best crop.
          </p>

          <div className="grid sm:grid-cols-2 gap-4">
            {[
              { k: 'N', l: 'Nitrogen (N) kg/ha', ph: '0-140' },
              { k: 'P', l: 'Phosphorus (P) kg/ha', ph: '5-145' },
              { k: 'K', l: 'Potassium (K) kg/ha', ph: '5-205' },
              { k: 'temperature', l: 'Temperature (°C)', ph: '10-45' },
              { k: 'humidity', l: 'Humidity (%)', ph: '14-100' },
              { k: 'ph', l: 'Soil pH', ph: '3.5-9.5', step: 0.1 },
              { k: 'rainfall', l: 'Rainfall (mm)', ph: '20-3000' },
            ].map(f => (
              <div key={f.k}>
                <label className="font-mono text-[11px] text-mid tracking-wider uppercase block mb-1">
                  {f.l}
                </label>
                <input
                  type="number"
                  value={fields[f.k]}
                  onChange={e => onChange(f.k, e.target.value)}
                  placeholder={f.ph}
                  step={f.step || 1}
                />
              </div>
            ))}
          </div>

          <label className="flex items-center gap-2 mt-5 text-sm text-mid cursor-pointer">
            <input
              type="checkbox"
              checked={useWeather}
              onChange={e => setUseWeather(e.target.checked)}
              className="accent-chrome"
            />
            <MapPin size={15} />
            Use live climate data (GPS)
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
                  <input
                    type="number"
                    value={lat}
                    onChange={e => setLat(parseFloat(e.target.value))}
                  />
                </div>
                <div>
                  <label className="font-mono text-[11px] text-mid">Longitude</label>
                  <input
                    type="number"
                    value={lon}
                    onChange={e => setLon(parseFloat(e.target.value))}
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <button
            onClick={submit}
            disabled={loading}
            className="leaf-btn w-full py-3.5 mt-6 font-heading text-base flex items-center justify-center gap-2"
          >
            {loading ? <><Loader2 className="animate-spin" size={18} /> Analysing...</> : '> Get Recommendation'}
          </button>
          {!loading && <div className="h-3.5" />}
        </div>

        {/* Results */}
        <div>
          <AnimatePresence mode="wait">
            {result && (
              <motion.div
                key="result"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="glass-card p-8"
              >
                <p className="font-mono text-[11px] text-chrome tracking-widest uppercase mb-4">
                  Recommended Crop
                </p>
                <p className="font-heading text-4xl font-bold text-gradient mb-2 capitalize">
                  {result.recommended_crop}
                </p>
                <p className="font-mono text-sm text-mid mb-5">
                  Confidence:{' '}
                  <span className="text-gradient font-bold">{result.confidence.toFixed(2)}%</span>
                </p>

                <div className="h-3 rounded-full bg-white/5 mb-6 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${result.confidence}%` }}
                    transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
                    className="h-full rounded-full bg-gradient-to-r from-chrome to-silver-mid"
                  />
                </div>

                <p className="text-mid font-body text-sm mb-5">{tip}</p>

                <p className="font-mono text-[11px] text-mid tracking-wider uppercase mb-3">
                  Top 3 Alternatives
                </p>
                <div className="space-y-3">
                  {result.top3?.map((t, i) => (
                    <div key={i} className="flex items-center gap-3">
                      <span className="font-mono text-xs text-muted w-5">#{i + 1}</span>
                      <div className="flex-1">
                        <div className="flex justify-between text-xs font-mono mb-0.5">
                          <span className="text-white-soft capitalize">{t.crop}</span>
                          <span className="text-chrome">{t.confidence.toFixed(2)}%</span>
                        </div>
                        <div className="h-2 rounded-full bg-white/5 overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${t.confidence}%` }}
                            transition={{ duration: 0.9, delay: i * 0.12 }}
                            className="h-full rounded-full bg-chrome/50"
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}

            {!result && !error && (
              <div className="glass-card p-12 flex flex-col items-center justify-center text-center min-h-[400px]">
                <Leaf className="text-muted mb-4 opacity-40" size={56} />
                <p className="text-muted italic font-body">Your recommendation will appear here</p>
              </div>
            )}

            {error && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="glass-card border-amber/40 p-6"
              >
                <p className="font-heading font-semibold text-amber mb-2">API Error</p>
                <p className="text-mid font-body text-sm mb-4">{error}</p>
                <p className="text-xs text-muted">Make sure the Flask server is running on localhost:5000</p>
              </motion.div>
            )}
          </AnimatePresence>

          <Link
            to="/"
            className="ghost-btn inline-flex items-center gap-2 px-5 py-2.5 mt-6 text-sm font-heading"
          >
            {'<-'} Back to Home
          </Link>
        </div>
      </div>
    </motion.div>
  );
}
