/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        'abyss':     '#030406',
        'deep':      '#06080c',
        'surface':   '#0c0f15',
        'elevated':  '#141820',
        'silver':    '#9ca3af',
        'silver-dim':'#6b7280',
        'silver-mid':'#c4cacd',
        'silver-hi': '#e5e8eb',
        'chrome':    '#b9c3cf',
        'white-soft':'#f3f4f6',
        'mid':       '#9ca3af',
        'muted':     '#6b7280',
        'leaf':      '#7dd3a8',
        'amber':     '#e2b96f',
      },
      fontFamily: {
        heading: ['Space Grotesk', 'sans-serif'],
        body: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'shimmer':    'shimmer 2s linear infinite',
        'pulse-glow': 'pulseGlow 2.2s ease-in-out infinite',
        'float':      'float 7s ease-in-out infinite',
        'scan':       'scan 4s ease-in-out infinite',
        'drift':      'drift 22s ease-in-out infinite',
      },
      keyframes: {
        shimmer: {
          '0%':   { backgroundPosition: '-200% center' },
          '100%': { backgroundPosition: '200% center' },
        },
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(185,195,207,0.3)' },
          '50%':      { boxShadow: '0 0 20px 5px rgba(185,195,207,0)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%':      { transform: 'translateY(-14px)' },
        },
        scan: {
          '0%':   { top: '0%', opacity: 0 },
          '5%':   { opacity: 0.85 },
          '92%':  { opacity: 0.85 },
          '100%': { top: '100%', opacity: 0 },
        },
        drift: {
          '0%, 100%': { transform: 'translate(0,0) rotate(0deg)' },
          '33%':      { transform: 'translate(28px,-18px) rotate(0.8deg)' },
          '66%':      { transform: 'translate(-18px,14px) rotate(-0.8deg)' },
        },
      },
    },
  },
  plugins: [],
};
