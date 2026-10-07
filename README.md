# AgriSense — AI-Based Smart Climate Adaptive Agriculture System

An end-to-end ML system for smart agriculture: **crop recommendation**, **yield
forecasting**, and **plant disease detection**, served by a Flask API with a
React desktop web app, a React-Native (Expo) mobile app, and a conversational
farm assistant.

---

## Current Status

All three models are trained, leak-free, and served. Metrics below are
**honest holdout numbers** measured after fixing a target-leakage bug and a
data-pipeline corruption bug (both documented further down).

| Module | Model | Accuracy / Score | Status |
|--------|-------|------------------|--------|
| Crop Recommendation | XGBoost (15 leak-free features) | **76.85% top-1**, **97.75% top-3**, 5-fold CV 76.23% | ✅ Deployed |
| Crop Recommendation (family mode) | XGBoost, 11 agro-climatic families | **80.42% top-1**, **99.53% top-3** | ✅ Deployed (optional endpoint) |
| Yield Forecasting | CNN-LSTM + Attention | **R² 0.980**, RMSE 1.19, MAPE 20.8% | ✅ Deployed |
| Disease Detection | EfficientNetV2B0 @ 224px, 5-view TTA | **97.85% top-1**, macro F1 0.976 | ✅ Deployed |
| Farm Assistant | NVIDIA NIM / Anthropic (tool-calling) | — | ✅ Deployed |

### Two bugs worth knowing about

1. **Target leakage (crop model).** The original feature pipeline derived
   season/soil/water/category features *from the crop label itself*, inflating
   accuracy to 98.2%. The honest, leak-free model scores **76.85% top-1**.
   Never reintroduce label-derived features.
2. **Corrupted disease pipeline.** `preprocess.py` rescaled images to `[0,1]`
   then applied `RandomBrightness` with its default `value_range=(0,255)`,
   shifting every image by ~51 units and capping accuracy at 66.9%. The
   corrected `[0,255]` pipeline reaches **97.85%**. The v6/v7 trainers use
   self-contained loaders and do **not** use `preprocess.load_disease_data`.

### Known issue

- The farm assistant can exceed the 60 s HTTP timeout on NVIDIA's
  `nemotron-3.5-lightning` model during cold start, which triggers the built-in
  demo-mode fallback. Swap `NVIDIA_MODEL` (see below) if this happens.

---

## Repository Layout

```
Mini Project/
├── app/
│   ├── app.py               # Flask API server (3 models + helpers)
│   ├── agent_bp.py          # Conversational agent blueprint (/api/chat)
│   └── templates/index.html
├── src/
│   ├── preprocess.py        # Data loading / preprocessing
│   ├── crop_features.py     # Leak-free feature engineering
│   ├── weather_client.py    # Open-Meteo API integration
│   ├── train_crop_v11.py    # XGBoost crop (leak-free)          ← deployed
│   ├── train_crop_family.py # XGBoost crop families             ← deployed
│   ├── train_yield_v2.py    # CNN-LSTM + Attention yield        ← deployed
│   ├── train_disease_v6.py  # EfficientNetV2B0 disease          ← deployed
│   └── train_disease_v7.py  # EfficientNetV2B1 + Mixup (experiment)
├── frontend/                # Desktop web app (Vite + React + Tailwind)
├── mobile-app/              # Mobile app (React Native / Expo)
├── models/                  # Trained artifacts (gitignored)
├── data/{raw,processed}/    # Datasets (gitignored)
├── outputs/                 # Metrics JSON + plots (gitignored)
├── notebooks/               # EDA only
├── requirements.txt
└── .env                     # Secrets (gitignored)
```

---

## Prerequisites

| Tool | Version used | Notes |
|------|--------------|-------|
| Python | 3.12 | TensorFlow + XGBoost |
| Node.js | 24.x | Frontend + mobile app |
| npm | 11.x | |
| Expo Go | latest | Physical phone testing (App Store / Play Store) |

---

## 1. Setup

```bash
git clone <your-repo-url>
cd "Mini Project"

# Python dependencies
pip install -r requirements.txt
```

### Environment variables

Copy the template and fill it in:

```bash
cp .env.example .env
```

| Variable | Purpose |
|----------|---------|
| `PORT` | Flask port (default `5000`) |
| `FLASK_DEBUG` | `1` to enable the reloader |
| `CORS_ORIGINS` | Comma-separated allowed origins |
| `NVIDIA_API_KEY` | NVIDIA NIM key (preferred agent provider) |
| `NVIDIA_BASE_URL` | Default `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_MODEL` | Tool-calling model (default `nvidia/nemotron-3.5-lightning-30b-a3b`) |
| `ANTHROPIC_API_KEY` | Optional fallback agent provider |

> **Agent provider:** NVIDIA is used when `NVIDIA_API_KEY` is set, otherwise
> Anthropic. If neither works, the agent replies in **demo mode** instead of
> erroring. Models verified to support tool-calling on NVIDIA NIM:
> `nvidia/nemotron-3.5-lightning-30b-a3b` (fast),
> `nvidia/nemotron-3-super-120b-a12b` (higher quality), `openai/gpt-oss-20b`.

### Datasets

Download into `data/raw/` (requires `~/.kaggle/kaggle.json`):

```bash
kaggle datasets download -d atharvaingle/crop-recommendation-dataset -p data/raw/ --unzip
kaggle datasets download -d patelris/crop-yield-prediction-dataset -p data/raw/ --unzip
kaggle datasets download -d emmarex/plantdisease -p data/raw/ --unzip
```

---

## 2. Training

Run from the repository root. Models write to `models/`, metrics to `outputs/`.

```bash
# Crop recommendation (leak-free) -> models/xgb_crop_v11.pkl
python src/train_crop_v11.py

# Crop families (11 agro-climatic groups) -> models/xgb_crop_family.pkl
python src/train_crop_family.py

# Yield forecasting -> models/cnn_lstm_yield_v2.h5
python src/train_yield_v2.py

# Disease detection -> models/efficientnet_disease_v6.h5
python src/train_disease_v6.py
```

---

## 3. Run the backend

```bash
python app/app.py
# -> http://localhost:5000
```

Serves on `0.0.0.0:5000`, so a phone on the same Wi-Fi can reach it at
`http://<your-lan-ip>:5000`.

Production-style:

```bash
gunicorn app.app:app
```

### API endpoints

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| `GET` | `/health` | — | `{status, models, models_loaded}` |
| `GET` | `/encoders` | — | Valid `areas` / `items` for yield |
| `POST` | `/predict-crop` | `{N,P,K,temperature,humidity,ph,rainfall,lat?,lon?}` | `{recommended_crop, confidence, top3[]}` |
| `POST` | `/predict-crop-family` | same as above | `{recommended_family, confidence, top3[]}` |
| `POST` | `/predict-yield` | `{area,crop,year,rainfall,pesticides,avg_temp,area_ha?,lat?,lon?}` | `{yield_per_ha, total_yield, unit}` |
| `POST` | `/predict-yield-raw` | `{area_code,item_code,...}` | same as above |
| `POST` | `/predict-disease` | multipart, file field `image` | `{disease_class, confidence, top3_classes[]}` |
| `POST` | `/api/chat` | `{messages:[{role,content}]}` | `{response}` or `{response, demo_mode:true}` |

Examples:

```bash
curl http://localhost:5000/health

curl -X POST http://localhost:5000/predict-crop \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":21,"humidity":82,"ph":6.5,"rainfall":202}'

curl -X POST http://localhost:5000/predict-yield \
  -H "Content-Type: application/json" \
  -d '{"area":"India","crop":"Rice, paddy","year":2024,"rainfall":1200,"pesticides":200,"avg_temp":27,"area_ha":2.5}'

curl -X POST http://localhost:5000/predict-disease -F "image=@leaf.jpg"

curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What should I plant with N=90, P=42, K=43, pH 6.5?"}]}'
```

---

## 4. Run the desktop web app

```bash
cd frontend
npm install
npm run dev
# -> http://localhost:5173
```

Other scripts:

```bash
npm run build     # production build -> frontend/dist
npm run preview   # preview the production build
npm run lint      # oxlint
```

Point it at a non-default API with `VITE_API_BASE`:

```bash
echo "VITE_API_BASE=http://localhost:5000" > .env.local
```

---

## 5. Run the mobile app

```bash
cd mobile-app
npm install
cp .env.example .env      # set EXPO_PUBLIC_API_URL
npm start                 # scan the QR with Expo Go
```

| Target | `EXPO_PUBLIC_API_URL` |
|--------|-----------------------|
| Physical phone (same Wi-Fi) | `http://<your-lan-ip>:5000` |
| Android emulator | `http://10.0.2.2:5000` |
| Web / same machine | `http://localhost:5000` |

The app has a **Mobile / Desktop** toggle (bottom-left). Mobile view is the
Android-Material UI; desktop view embeds the desktop web app from
`EXPO_PUBLIC_FRONTEND_URL` (default `http://localhost:5173`). Run the frontend
dev server for desktop view to render. A chat FAB sits bottom-right in both
modes.

> **Speech-to-text:** `expo-speech` is text-to-speech only. True on-device STT
> needs a development build, so the mic uses the phone's keyboard dictation
> (works in Expo Go). Read-aloud uses `expo-speech`.

---

## 6. Run everything together

Three terminals, from the repository root:

```bash
# Terminal 1 — backend (required)
python app/app.py
```

```bash
# Terminal 2 — desktop web app (required for desktop view)
cd frontend && npm run dev
```

```bash
# Terminal 3 — mobile app
cd mobile-app && npm start
```

---

## Tech Stack

- **ML:** TensorFlow / Keras, XGBoost, scikit-learn
- **Backend:** Flask, Flask-CORS, python-dotenv, requests
- **Desktop web:** React 19, Vite, Tailwind CSS, Framer Motion, React Router
- **Mobile:** React Native 0.76, Expo SDK 52, React Navigation
- **Agent:** NVIDIA NIM / Anthropic (OpenAI-compatible tool calling)
- **External APIs:** Open-Meteo (live climate)

---

## Deploy

```bash
gunicorn app.app:app
# Render / Railway / any WSGI host
```

Set `CORS_ORIGINS` to your deployed frontend URL, and set the agent API key in
the host's environment (never in the repo).
