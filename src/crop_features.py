import numpy as np
import pandas as pd

# ── Crop → feature lookup (built from agronomic knowledge) ──────────────────
# Season: 0=Kharif, 1=Rabi, 2=Summer, 3=Perennial
SEASON = {
    'rice': 0, 'maize': 0, 'cotton': 0, 'jute': 0, 'sugarcane': 0,
    'soybean': 0, 'groundnut': 0, 'mothbeans': 0, 'mungbean': 0,
    'pigeonpeas': 0, 'tapioca': 0, 'turmeric': 0, 'ginger': 0,
    'blackgram': 0, 'muskmelon': 2, 'watermelon': 2,
    'wheat': 1, 'barley': 1, 'mustard': 1, 'onion': 1, 'potato': 1,
    'chickpea': 1, 'lentil': 1, 'sunflower': 1, 'coriander': 1,
    'cabbage': 1, 'okra': 0, 'brinjal': 0, 'tomato': 0,
    'chilli': 0, 'banana': 3, 'coconut': 3, 'coffee': 3,
    'tea': 3, 'rubber': 3, 'mango': 3, 'papaya': 3, 'orange': 3,
    'apple': 3, 'grapes': 3, 'pomegranate': 3, 'cashew': 3,
    'cardamom': 3, 'blackpepper': 3, 'tobacco': 0, 'sweetpotato': 0,
}

# Soil: 0=Alluvial, 1=Black, 2=Red, 3=Loamy, 4=Sandy, 5=Laterite, 6=Mountain, 7=Clay
SOIL = {
    'rice': 0, 'wheat': 0, 'barley': 0, 'mustard': 0, 'onion': 0,
    'maize': 1, 'cotton': 1, 'groundnut': 1, 'jute': 0,
    'soybean': 3, 'sunflower': 2, 'potato': 3, 'cabbage': 3,
    'chickpea': 1, 'lentil': 1, 'pigeonpeas': 1, 'mothbeans': 4,
    'mungbean': 3, 'blackgram': 3, 'tapioca': 2, 'sweetpotato': 3,
    'turmeric': 3, 'ginger': 3, 'coriander': 0, 'okra': 3,
    'brinjal': 3, 'tomato': 3, 'chilli': 3, 'tobacco': 3,
    'banana': 3, 'coconut': 5, 'coffee': 5, 'tea': 6, 'rubber': 5,
    'mango': 2, 'papaya': 3, 'orange': 2, 'apple': 6,
    'grapes': 2, 'pomegranate': 2, 'cashew': 5, 'cardamom': 6,
    'blackpepper': 5, 'sugarcane': 3, 'muskmelon': 3, 'watermelon': 3,
}

# Water req: 0=Low, 1=Medium, 2=High, 3=Very High
WATER = {
    'wheat': 0, 'barley': 0, 'chickpea': 0, 'lentil': 0, 'mustard': 0,
    'groundnut': 0, 'mothbeans': 0, 'sunflower': 1,
    'soybean': 1, 'maize': 1, 'cotton': 1, 'jute': 2,
    'pigeonpeas': 0, 'mungbean': 0, 'blackgram': 0,
    'onion': 1, 'cabbage': 2, 'potato': 2, 'coriander': 1,
    'brinjal': 2, 'tomato': 2, 'chilli': 1, 'okra': 2,
    'tobacco': 1, 'turmeric': 2, 'ginger': 2, 'sweetpotato': 1,
    'tapioca': 0, 'rice': 3, 'sugarcane': 3,
    'banana': 3, 'coconut': 2, 'watermelon': 2, 'muskmelon': 2,
    'mango': 1, 'papaya': 1, 'orange': 1, 'apple': 1,
    'grapes': 1, 'pomegranate': 0, 'cashew': 0,
    'tea': 3, 'coffee': 3, 'rubber': 3,
    'cardamom': 3, 'blackpepper': 3,
}

# Nitrogen fixer: legumes fix atmospheric N
NITROGEN_FIX = {
    'chickpea': 1, 'lentil': 1, 'soybean': 1, 'groundnut': 1,
    'pigeonpeas': 1, 'mothbeans': 1, 'mungbean': 1, 'blackgram': 1,
    # non-legumes
    'rice': 0, 'wheat': 0, 'barley': 0, 'maize': 0, 'cotton': 0,
    'jute': 0, 'sugarcane': 0, 'sunflower': 0, 'mustard': 0,
    'onion': 0, 'potato': 0, 'cabbage': 0, 'tomato': 0,
    'brinjal': 0, 'chilli': 0, 'okra': 0, 'tobacco': 0,
    'tapioca': 0, 'turmeric': 0, 'ginger': 0, 'sweetpotato': 0,
    'coriander': 0, 'banana': 0, 'coconut': 0,
    'watermelon': 0, 'muskmelon': 0,
    'mango': 0, 'papaya': 0, 'orange': 0, 'apple': 0,
    'grapes': 0, 'pomegranate': 0, 'cashew': 0,
    'tea': 0, 'coffee': 0, 'rubber': 0,
    'cardamom': 0, 'blackpepper': 0,
}

# Category: 0=Cereal, 1=Pulse, 2=Vegetable, 3=Fruit, 4=Spice,
#           5=Fiber, 6=Oilseed, 7=Beverage, 8=Tuber, 9=Commercial
CATEGORY = {
    'rice': 0, 'wheat': 0, 'barley': 0, 'maize': 0,
    'chickpea': 1, 'lentil': 1, 'pigeonpeas': 1, 'mothbeans': 1,
    'mungbean': 1, 'blackgram': 1, 'soybean': 1, 'groundnut': 6,
    'cabbage': 2, 'tomato': 2, 'brinjal': 2, 'onion': 2,
    'potato': 8, 'okra': 2, 'chilli': 4, 'sweetpotato': 8,
    'tapioca': 8, 'coriander': 4, 'watermelon': 3, 'muskmelon': 3,
    'turmeric': 4, 'ginger': 4, 'tobacco': 9,
    'cotton': 5, 'jute': 5, 'sugarcane': 9,
    'banana': 3, 'mango': 3, 'papaya': 3, 'orange': 3,
    'apple': 3, 'grapes': 3, 'pomegranate': 3, 'cashew': 3,
    'coconut': 3, 'mustard': 6, 'sunflower': 6,
    'tea': 7, 'coffee': 7, 'rubber': 5,
    'cardamom': 4, 'blackpepper': 4,
}

# Growth period (days): approx median for annuals, 365 for perennials
GROWTH_DAYS = {
    'rice': 120, 'wheat': 120, 'barley': 100, 'maize': 110,
    'chickpea': 100, 'lentil': 100, 'pigeonpeas': 120, 'mothbeans': 80,
    'mungbean': 70, 'blackgram': 90, 'soybean': 110, 'groundnut': 120,
    'cabbage': 90, 'tomato': 90, 'brinjal': 100, 'onion': 120,
    'potato': 100, 'okra': 60, 'chilli': 90, 'sweetpotato': 120,
    'tapioca': 300, 'coriander': 60, 'watermelon': 90, 'muskmelon': 90,
    'turmeric': 270, 'ginger': 270, 'tobacco': 120,
    'cotton': 160, 'jute': 120, 'sugarcane': 365,
    'banana': 365, 'mango': 365, 'papaya': 365, 'orange': 365,
    'apple': 365, 'grapes': 365, 'pomegranate': 365, 'cashew': 365,
    'coconut': 365, 'mustard': 90, 'sunflower': 100,
    'tea': 365, 'coffee': 365, 'rubber': 365,
    'cardamom': 365, 'blackpepper': 365,
}


def add_features(df):
    """Add engineered features to the crop dataframe."""
    names = df['label'].values

    n = len(df)
    season = np.array([SEASON.get(c, 0) for c in names], dtype=np.int32)
    soil = np.array([SOIL.get(c, 3) for c in names], dtype=np.int32)
    water = np.array([WATER.get(c, 1) for c in names], dtype=np.int32)
    nfix = np.array([NITROGEN_FIX.get(c, 0) for c in names], dtype=np.int32)
    cat = np.array([CATEGORY.get(c, 2) for c in names], dtype=np.int32)
    grow = np.array([GROWTH_DAYS.get(c, 100) for c in names], dtype=np.float32)
    grow_norm = (grow - 60) / 350  # normalize to ~0-1

    # Derived ratios from existing features (df already has N,P,K,temperature,humidity,ph,rainfall)
    N = df['N'].values.astype(np.float32)
    P = df['P'].values.astype(np.float32)
    K = df['K'].values.astype(np.float32)
    T = df['temperature'].values.astype(np.float32)
    H = df['humidity'].values.astype(np.float32)
    rain = df['rainfall'].values.astype(np.float32)
    pH = df['ph'].values.astype(np.float32)

    n_pk_total = N + P + K + 1
    N_pct = N / n_pk_total
    P_pct = P / n_pk_total
    K_pct = K / n_pk_total
    n_p_ratio = np.where(P > 0, N / P, 0)
    n_k_ratio = np.where(K > 0, N / K, 0)
    p_k_ratio = np.where(K > 0, P / K, 0)
    hum_temp = H * T / 100
    rain_ph = rain * pH / 100

    # Combine: original 7 + 12 new = 19 features
    feat_names = [
        'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
        'season', 'soil_type', 'water_req', 'n_fix', 'category', 'growth_days',
        'N_pct', 'P_pct', 'K_pct', 'N_P_ratio', 'N_K_ratio', 'P_K_ratio',
        'hum_temp', 'rain_ph',
    ]

    X = np.column_stack([
        N, P, K, T, H, pH, rain,
        season, soil, water, nfix, cat, grow_norm,
        N_pct, P_pct, K_pct, n_p_ratio, n_k_ratio, p_k_ratio,
        hum_temp, rain_ph,
    ]).astype(np.float32)

    return X, feat_names


def build_feature_vector(N, P, K, temperature, humidity, ph, rainfall,
                         season=0, soil_type=3, water_req=1, n_fix=0,
                         category=2, growth_days=100):
    n_pk_total = N + P + K + 1
    N_pct = N / n_pk_total
    P_pct = P / n_pk_total
    K_pct = K / n_pk_total
    n_p_ratio = N / P if P > 0 else 0
    n_k_ratio = N / K if K > 0 else 0
    p_k_ratio = P / K if K > 0 else 0
    hum_temp = humidity * temperature / 100
    rain_ph = rainfall * ph / 100
    grow_norm = (growth_days - 60) / 350
    return np.array([[N, P, K, temperature, humidity, ph, rainfall,
                      season, soil_type, water_req, n_fix, category, grow_norm,
                      N_pct, P_pct, K_pct, n_p_ratio, n_k_ratio, p_k_ratio,
                      hum_temp, rain_ph]], dtype=np.float32)


# Crop name → all derived features (for auto-fill in API)
CROP_FEATURES = {
    'rice':         (0, 0, 3, 0, 0, 120),
    'maize':        (0, 1, 1, 0, 0, 110),
    'wheat':        (1, 0, 0, 0, 0, 120),
    'barley':       (1, 0, 0, 0, 0, 100),
    'chickpea':     (1, 1, 0, 1, 1, 100),
    'kidneybeans':  (0, 3, 1, 1, 1, 100),
    'pigeonpeas':   (0, 1, 0, 1, 1, 120),
    'mothbeans':    (0, 4, 0, 1, 1, 80),
    'mungbean':     (0, 3, 0, 1, 1, 70),
    'blackgram':    (0, 3, 0, 1, 1, 90),
    'lentil':       (1, 1, 0, 1, 1, 100),
    'jute':         (0, 0, 2, 0, 5, 120),
    'cotton':       (0, 1, 1, 0, 5, 160),
    'sugarcane':    (0, 3, 3, 0, 9, 365),
    'sunflower':    (1, 2, 1, 0, 6, 100),
    'soybean':      (0, 3, 1, 0, 6, 110),
    'groundnut':    (0, 1, 0, 0, 6, 120),
    'mustard':      (1, 1, 0, 0, 6, 90),
    'potato':       (1, 3, 2, 0, 8, 100),
    'onion':        (1, 0, 1, 0, 2, 120),
    'cabbage':      (1, 3, 2, 0, 2, 90),
    'tomato':       (0, 3, 2, 0, 2, 90),
    'brinjal':      (0, 3, 2, 0, 2, 100),
    'chilli':       (0, 3, 1, 0, 4, 90),
    'okra':         (0, 3, 2, 0, 2, 60),
    'coriander':    (1, 0, 1, 0, 4, 60),
    'tobacco':      (0, 3, 1, 0, 9, 120),
    'turmeric':     (0, 3, 2, 0, 4, 270),
    'ginger':       (0, 3, 2, 0, 4, 270),
    'sweetpotato':  (0, 3, 1, 0, 8, 120),
    'tapioca':      (0, 2, 0, 0, 8, 300),
    'banana':       (3, 3, 3, 0, 3, 365),
    'mango':        (3, 2, 1, 0, 3, 365),
    'papaya':       (3, 3, 1, 0, 3, 365),
    'orange':       (3, 2, 1, 0, 3, 365),
    'apple':        (3, 6, 1, 0, 3, 365),
    'grapes':       (3, 2, 1, 0, 3, 365),
    'pomegranate':  (3, 2, 0, 0, 3, 365),
    'cashew':       (3, 5, 0, 0, 3, 365),
    'coconut':      (3, 5, 2, 0, 3, 365),
    'tea':          (3, 6, 3, 0, 7, 365),
    'coffee':       (3, 5, 3, 0, 7, 365),
    'rubber':       (3, 5, 3, 0, 5, 365),
    'cardamom':     (3, 6, 3, 0, 4, 365),
    'blackpepper':  (3, 5, 3, 0, 4, 365),
    'watermelon':   (2, 3, 2, 0, 3, 90),
    'muskmelon':    (2, 3, 2, 0, 3, 90),
}


def get_crop_features(crop_name):
    """Return (season, soil_type, water_req, n_fix, category, growth_days) for a crop."""
    return CROP_FEATURES.get(crop_name.lower().strip(), None)


# ──────────────────────────────────────────────────────────────────────────
# LEAK-FREE feature engineering for crop RECOMMENDATION.
# Uses ONLY the 7 measurable inputs (N, P, K, temperature, humidity, ph,
# rainfall). No label-derived categorical features — those caused target
# leakage in v10 (a 6-feature label-lookup alone reached 93.7% CV).
# ──────────────────────────────────────────────────────────────────────────
LEGIT_FEATURES = [
    'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
    'N_pct', 'P_pct', 'K_pct', 'N_P_ratio', 'N_K_ratio', 'P_K_ratio',
    'hum_temp', 'rain_ph', 'npk_total', 'log_N', 'log_P', 'log_K',
    'temp_rain', 'ph_dev', 'humidity_rain',
]


def compute_legit_features(N, P, K, T, H, ph, rain):
    """Leak-free features from the 7 raw inputs. Accepts scalars or arrays,
    returns an (n, 22) float32 matrix in LEGIT_FEATURES order."""
    N = np.asarray(N, dtype=np.float32); P = np.asarray(P, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32); T = np.asarray(T, dtype=np.float32)
    H = np.asarray(H, dtype=np.float32); ph = np.asarray(ph, dtype=np.float32)
    rain = np.asarray(rain, dtype=np.float32)
    tot = N + P + K + 1
    N_pct = N / tot; P_pct = P / tot; K_pct = K / tot
    n_p = np.where(P > 0, N / P, 0)
    n_k = np.where(K > 0, N / K, 0)
    p_k = np.where(K > 0, P / K, 0)
    hum_temp = H * T / 100
    rain_ph = rain * ph / 100
    npk_total = N + P + K
    log_N = np.log1p(N); log_P = np.log1p(P); log_K = np.log1p(K)
    temp_rain = T * rain / 100
    ph_dev = np.abs(ph - 6.5)
    humidity_rain = H * rain / 1000
    return np.column_stack([
        N, P, K, T, H, ph, rain,
        N_pct, P_pct, K_pct, n_p, n_k, p_k, hum_temp, rain_ph,
        npk_total, log_N, log_P, log_K, temp_rain, ph_dev, humidity_rain,
    ]).astype(np.float32)


def build_feature_vector_honest(N, P, K, temperature, humidity, ph, rainfall):
    """Single-row leak-free feature vector for the recommendation API."""
    return compute_legit_features(N, P, K, temperature, humidity, ph, rainfall)