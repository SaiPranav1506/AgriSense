# AgriSense — AI-Based Smart Climate Adaptive Agriculture System

An end-to-end machine learning system for smart agriculture with three core capabilities: crop recommendation, yield forecasting, and plant disease detection.

## Modules

| Module | Model | Dataset | Expected Accuracy |
|--------|-------|---------|-------------------|
| Crop Recommendation | XGBoost | Crop Recommendation (Kaggle) — 2,200 rows, 22 crops | 93–98% |
| Yield Forecasting | DNN with Categorical Embeddings | Crop Yield Prediction (Kaggle) — 28,242 rows | R² ≈ 0.97–0.98 |
| Disease Detection | MobileNetV2 (Transfer Learning) | PlantVillage (Kaggle) — 41K images, 15 classes | 94–97% |

## Folder Structure

```
Mini Project/
├── data/
│   ├── raw/          # Original datasets (gitignored)
│   └── processed/    # Preprocessed sequences (gitignored)
├── src/
│   ├── preprocess.py      # Data loading & preprocessing
│   ├── train_crop.py      # XGBoost crop recommendation
│   ├── train_yield.py     # DNN yield forecasting
│   ├── train_disease.py   # MobileNetV2 disease detection
│   └── weather_client.py  # Open-Meteo API integration
├── models/          # Trained model files (gitignored)
├── notebooks/
│   └── 01_EDA.ipynb # Exploratory data analysis
├── app/
│   ├── app.py       # Flask API server
│   └── templates/
│       └── index.html
├── outputs/         # Plots & visualizations (gitignored)
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Setup

Download the three datasets from Kaggle (requires Kaggle API key in `~/.kaggle/kaggle.json`):

```bash
# Crop Recommendation
kaggle datasets download -d atharvaingle/crop-recommendation-dataset -p data/raw/ --unzip

# Crop Yield Prediction
kaggle datasets download -d patelris/crop-yield-prediction-dataset -p data/raw/ --unzip

# Plant Disease
kaggle datasets download -d emmarex/plantdisease -p data/raw/ --unzip
```

## Training

```bash
# Preprocessing
python src/preprocess.py

# Train all models
python src/train_crop.py      # XGBoost → models/xgb_crop.pkl
python src/train_yield.py     # DNN        → models/cnn_lstm_yield.h5
python src/train_disease.py   # MobileNetV2 → models/mobilenet_disease.h5
```

## Run Locally

```bash
python app/app.py
# → http://localhost:5000
```

### API Endpoints

**Crop Recommendation:** `POST /predict-crop`
```json
{"N":80, "P":42, "K":43, "temperature":27, "humidity":68, "ph":6.5, "rainfall":120}
```

**Yield Forecast:** `POST /predict-yield`
```json
{"area":"India", "crop":"Rice", "year":2024, "rainfall":1200, "pesticides":200, "avg_temp":27}
```

**Disease Detection:** `POST /predict-disease` (multipart form with `image` file)

## Deploy

```bash
gunicorn app.app:app
# Deploy on Render.com, Railway, or any WSGI-compatible host
```