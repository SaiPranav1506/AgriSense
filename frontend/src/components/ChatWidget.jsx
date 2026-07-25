import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageCircle, X, Send, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import toast from 'react-hot-toast';
import { API_BASE } from '../config';

const WELCOME = `👋 Hello! I'm your **AgriSense farm assistant**.

Ask me about:
- 🌱 **What crop to plant** — tell me your soil N, P, K, and pH
- 📈 **Yield forecasts** — for your crop and farm size
- 🌤️ **Weather advice** — for your location

Just type your question below!`;

export default function ChatWidget() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([{ role: 'assistant', text: WELCOME }]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const msgEnd = useRef(null);
  const inpRef = useRef(null);

  useEffect(() => { msgEnd.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);
  useEffect(() => { if (open) setTimeout(() => inpRef.current?.focus(), 200); }, [open]);

  const send = async () => {
    const txt = input.trim();
    if (!txt || loading) return;
    setInput('');
    setMessages(p => [...p, { role: 'user', text: txt }]);
    setLoading(true);
    try {
      const r = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: [{ role: 'user', content: txt }] }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
      setMessages(p => [...p, { role: 'assistant', text: data.response }]);
    } catch (e) {
      toast.error('Agent unavailable — check server & API key');
      setMessages(p => [...p, {
        role: 'assistant',
        text: `⚠️ **Could not reach the farm assistant.**\n\n**Reason:** ${e.message}\n\nMake sure Flask on port 5000 has \`ANTHROPIC_API_KEY\` set and is running.`,
      }]);
    } finally { setLoading(false); }
  };

  const keyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <>
      {/* ── FAB button ── */}
      <button
        onClick={() => setOpen(o => !o)}
        className="fixed bottom-6 right-6 z-50 leaf-btn w-14 h-14 flex items-center justify-center rounded-full shadow-2xl shadow-black/40 hover:shadow-[0_0_28px_rgba(156,163,175,0.35)]"
        aria-label={open ? 'Close chat' : 'Open farm assistant'}
      >
        {open ? <X size={22} /> : <MessageCircle size={22} />}
      </button>

      {/* ── Chat panel ── */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 24, scale: 0.93 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 24, scale: 0.93 }}
            transition={{ duration: 0.22, ease: 'easeOut' }}
            className="fixed bottom-24 right-6 z-50 w-[380px] max-w-[calc(100vw-32px)]"
          >
            <div
              className="glass-card flex flex-col overflow-hidden"
              style={{ maxHeight: '540px', minHeight: '320px' }}
            >
              {/* Header */}
              <div className="flex items-center justify-between px-4 py-3 border-b border-silver/10 shrink-0">
                <div className="flex items-center gap-2.5">
                  <span className="relative flex h-2.5 w-2.5">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-silver opacity-50" />
                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-silver" />
                  </span>
                  <span className="font-heading font-semibold text-sm text-white-soft">Farm Assistant</span>
                </div>
                <span className="font-mono text-[10px] text-muted tracking-wider">CLAUDE AI</span>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3" style={{ minHeight: '180px' }}>
                {messages.map((m, i) => (
                  <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[88%] rounded-xl px-3.5 py-2.5 ${
                      m.role === 'user'
                        ? 'bg-chrome/15 text-white-soft rounded-tr-sm'
                        : 'bg-silver/[0.06] text-white-soft rounded-tl-sm border border-silver/[0.06]'
                    }`}>
                      {m.role === 'assistant' ? (
                        <div className="prose prose-invert prose-sm max-w-none font-body text-[13px] leading-relaxed [&_strong]:text-white-soft [&_code]:text-chrome [&_code]:bg-silver/10 [&_code]:px-1 [&_code]:rounded [&_p]:text-silver-mid [&_li]:text-silver-mid">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown>
                        </div>
                      ) : (
                        <p className="font-body text-[13px] leading-relaxed text-silver-mid">{m.text}</p>
                      )}
                    </div>
                  </div>
                ))}

                {loading && (
                  <div className="flex justify-start">
                    <div className="bg-silver/[0.06] border border-silver/[0.06] rounded-xl rounded-tl-sm px-4 py-3">
                      <div className="flex gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-silver-mid animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="w-2 h-2 rounded-full bg-silver-mid animate-bounce" style={{ animationDelay: '160ms' }} />
                        <span className="w-2 h-2 rounded-full bg-silver-mid animate-bounce" style={{ animationDelay: '320ms' }} />
                      </div>
                    </div>
                  </div>
                )}
                <div ref={msgEnd} />
              </div>

              {/* Input */}
              <div className="shrink-0 px-4 py-3 border-t border-silver/10">
                <div className="flex gap-2">
                  <input
                    ref={inpRef}
                    type="text"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={keyDown}
                    placeholder="Ask about your farm..."
                    disabled={loading}
                    className="flex-1 !bg-abyss/60 !border-silver/10 !text-sm !py-2.5 !rounded-lg"
                  />
                  <button
                    onClick={send}
                    disabled={!input.trim() || loading}
                    className="leaf-btn p-2.5 flex items-center justify-center shrink-0 !rounded-lg"
                    style={{ width: '40px', height: '40px' }}
                    aria-label="Send"
                  >
                    {loading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}