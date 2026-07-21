import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { Leaf, Loader2, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { API_BASE } from '../config';

function inferSpecies(className) {
  const name = (className || '').toLowerCase();
  if (name.includes('pepper')) return 'Pepper';
  if (name.includes('potato')) return 'Potato';
  if (name.includes('tomato')) return 'Tomato';
  return 'Plant';
}

function isHealthy(className) {
  return (className || '').toLowerCase().includes('healthy');
}

export default function DiseasePage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = useCallback(accepted => {
    const f = accepted[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp'] },
    maxFiles: 1,
  });

  const submit = async () => {
    if (!file) return;
    setScanning(true);
    setResult(null);
    setError(null);
    try {
      const fd = new FormData();
      fd.append('image', file);
      const r = await fetch(`${API_BASE}/predict-disease`, {
        method: 'POST',
        body: fd,
      });
      if (!r.ok) throw new Error((await r.json()).error || r.statusText);
      setResult(await r.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setScanning(false);
    }
  };

  const cls = result?.disease_class || '';
  const healthy = isHealthy(cls);
  const species = inferSpecies(cls);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.4, ease: 'easeInOut' }}
      className="min-h-screen pt-28 pb-16 px-6"
    >
      <div className="max-w-6xl mx-auto grid lg:grid-cols-2 gap-10">
        {/* Upload */}
        <div className="glass-card p-8">
          <div className="flex items-center gap-3 mb-2">
            <Leaf className="text-chrome" size={26} />
            <h2 className="font-heading text-2xl font-bold text-white-soft">
              Plant Disease Detection
            </h2>
          </div>
          <p className="text-mid font-body text-sm mb-6">
            Upload a leaf photo for AI-powered disease diagnosis.
          </p>

          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${
              isDragActive
                ? 'border-chrome bg-chrome/10'
                : 'border-chrome/20 hover:border-chrome/60 hover:bg-white/[0.02]'
            }`}
          >
            <input {...getInputProps()} />
            <Leaf className="mx-auto text-muted mb-3" size={44} />
            <p className="font-body text-white-soft text-sm font-medium">
              {isDragActive ? 'Drop the image here...' : 'Drop a leaf photo, or click to browse'}
            </p>
            <p className="text-muted text-xs mt-1 font-mono">Accepts JPG, PNG, WEBP</p>
          </div>

          {preview && (
            <div className="mt-5 flex flex-col items-center">
              <div className="relative rounded-xl overflow-hidden border border-chrome/20">
                <img
                  src={preview}
                  alt="preview"
                  className="max-h-60 object-contain bg-surface"
                />
                {scanning && (
                  <div className="absolute inset-0 pointer-events-none overflow-hidden">
                    <motion.div
                      initial={{ top: 0 }}
                      animate={{ top: '100%' }}
                      transition={{ duration: 1.8, repeat: Infinity, ease: 'linear' }}
                      className="absolute left-0 right-0 h-1 bg-chrome/70"
                      style={{ boxShadow: '0 0 12px rgba(99,102,241,0.6)' }}
                    />
                  </div>
                )}
              </div>
              <p className="font-mono text-[11px] text-mid mt-2">
                {file.name} · {(file.size / 1024).toFixed(1)} KB
              </p>
            </div>
          )}

          <button
            onClick={submit}
            disabled={!file || scanning}
            className="leaf-btn w-full py-3.5 mt-6 font-heading text-base flex items-center justify-center gap-2"
          >
            {scanning ? <><Loader2 className="animate-spin" size={18} /> Analysing...</> : '🔬 Analyse Leaf'}
          </button>
          {!scanning && <div className="h-3.5" />}
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
                  Diagnosis Result
                </p>

                <div className="flex items-start gap-3 mb-3">
                  {healthy ? (
                    <CheckCircle2 className="text-leaf shrink-0" size={26} />
                  ) : (
                    <AlertTriangle className="text-amber shrink-0" size={26} />
                  )}
                  <p
                    className={`font-heading text-3xl font-bold capitalize ${
                      healthy ? 'text-leaf' : 'text-amber'
                    }`}
                  >
                    {cls}
                  </p>
                </div>

                <p className="font-mono text-sm text-mid mb-6">
                  Confidence:{' '}
                  <span className={`font-bold ${healthy ? 'text-leaf' : 'text-amber'}`}>
                    {result.confidence?.toFixed(2)}%
                  </span>
                </p>

                <div className="h-3 rounded-full bg-white/5 mb-6 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${result.confidence}%` }}
                    transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
                    className={`h-full rounded-full ${healthy ? 'bg-gradient-to-r from-leaf to-chrome/60' : 'bg-gradient-to-r from-amber to-amber-light'}`}
                  />
                </div>

                {/* Advisory */}
                <div className={`p-4 rounded-lg border ${healthy ? 'border-leaf/30 bg-leaf/5' : 'border-amber/30 bg-amber/5'}`}>
                  <p className="font-body text-sm text-white-soft">
                    {healthy ? (
                      <>✅ Your <em>{species}</em> plant appears healthy. Continue regular monitoring.</>
                    ) : (
                      <>
                        ⚠️ <em>{cls}</em> detected on this {species} leaf. Apply targeted treatment
                        and isolate affected plants. Consult a local agronomist if spreading.
                      </>
                    )}
                  </p>
                </div>

                {/* Top 3 */}
                {result.top3_classes && (
                  <div className="mt-6">
                    <p className="font-mono text-[11px] text-mid tracking-wider uppercase mb-3">
                      Top 3 Predictions
                    </p>
                    <div className="space-y-2.5">
                      {result.top3_classes.map((t, i) => (
                        <div key={i} className="flex items-center gap-3">
                          <span className="font-mono text-xs text-muted w-5">#{i + 1}</span>
                          <div className="flex-1">
                            <div className="flex justify-between text-xs font-mono mb-0.5">
                              <span className="text-white-soft capitalize">{t.class}</span>
                              <span className="text-chrome">{t.confidence.toFixed(2)}%</span>
                            </div>
                            <div className="h-2 rounded-full bg-white/5 overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${t.confidence}%` }}
                                transition={{ duration: 0.9, delay: i * 0.1 }}
                                className="h-full rounded-full bg-chrome/50"
                              />
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            )}

            {!result && !error && (
              <div className="glass-card p-12 flex flex-col items-center justify-center text-center min-h-[400px]">
                <Leaf className="text-muted mb-4 opacity-40" size={56} />
                <p className="text-muted italic font-body">Upload a photo to begin diagnosis</p>
              </div>
            )}

            {error && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-card border-amber/40 p-6">
                <p className="font-heading font-semibold text-amber mb-2">⚠ API Error</p>
                <p className="text-mid text-sm mb-2">{error}</p>
                <p className="text-xs text-muted">Is Flask running?</p>
              </motion.div>
            )}
          </AnimatePresence>

          <Link
            to="/"
            className="ghost-btn inline-flex items-center gap-2 px-5 py-2.5 mt-6 text-sm font-heading"
          >
            ← Back to Home
          </Link>
        </div>
      </div>
    </motion.div>
  );
}
