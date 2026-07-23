import sys, os, io, pickle, logging
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(BASE, 'src'))
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
from weather_client import fetch_climate

# ── Logging ──
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── CORS ──
allowed_origins = os.environ.get(
    'CORS_ORIGINS',
    'http://localhost:9999,http://localhost:5173,http://localhost:3000'
).split(',')
CORS(app, origins=allowed_origins)

# ── Upload limits ──
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# ── Model paths ──
def _m(path):
    return os.path.join(BASE, path)

log.info("Loading models...")
crop_model    = pickle.load(open(_m('models/xgb_crop.pkl'),              'rb'))
yield_model   = load_model(_m('models/cnn_lstm_yield.h5'), compile=False)
disease_model = load_model(_m('models/mobilenet_disease.h5'), compile=False)
label_encoder = pickle.load(open(_m('models/label_encoder.pkl'),         'rb'))
scaler        = pickle.load(open(_m('models/scaler.pkl'),                'rb'))
y_scaler      = pickle.load(open(_m('models/y_scaler.pkl'),              'rb'))
class_names   = pickle.load(open(_m('models/disease_class_names.pkl'),   'rb'))
area_encoder  = pickle.load(open(_m('models/area_encoder.pkl'),          'rb'))
item_encoder  = pickle.load(open(_m('models/item_encoder.pkl'),          'rb'))
num_scaler    = pickle.load(open(_m('models/num_scaler.pkl'),            'rb'))
log.info("All models loaded.")


# ── Helpers ──
def _err(msg, code=400):
    return jsonify({'error': str(msg)}), code


# ── Routes ──
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'models': ['crop', 'yield', 'disease'],
        'models_loaded': all([
            crop_model, yield_model, disease_model,
            label_encoder, scaler, y_scaler, class_names,
            area_encoder, item_encoder, num_scaler,
        ])
    })


@app.route('/predict-crop', methods=['POST'])
def predict_crop():
    try:
        d = request.get_json()
        if not d:
            return _err('Request body must be JSON')
        N    = float(d.get('N', 0))
        P    = float(d.get('P', 0))
        K    = float(d.get('K', 0))
        temp = float(d.get('temperature', 25))
        hum  = float(d.get('humidity', 50))
        ph   = float(d.get('ph', 7.0))
        rain = float(d.get('rainfall', 100))
        if not (0 <= N <= 200 and 0 <= P <= 200 and 0 <= K <= 200):
            return _err('N, P, K must be in range 0–200')
        if not (3.5 <= ph <= 9.5):
            return _err('pH must be in range 3.5–9.5')
        if 'lat' in d and 'lon' in d:
            climate = fetch_climate(d['lat'], d['lon'])
            if climate is not None:
                temp = float(np.mean(climate[:, 0]))
                hum  = float(np.mean(climate[:, 1]))
        X_scaled = scaler.transform([[N, P, K, temp, hum, ph, rain]])
        proba    = crop_model.predict_proba(X_scaled)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        return jsonify({
            'recommended_crop': label_encoder.classes_[top3_idx[0]],
            'confidence': round(float(proba[top3_idx[0]] * 100), 2),
            'top3': [{'crop': label_encoder.classes_[i],
                      'confidence': round(float(proba[i] * 100), 2)}
                     for i in top3_idx]
        })
    except Exception as e:
        log.exception('predict-crop failed')
        return _err(str(e))


@app.route('/predict-yield', methods=['POST'])
def predict_yield():
    try:
        d = request.get_json()
        if not d:
            return _err('Request body must be JSON')
        area = d.get('area', 'India')
        item = d.get('crop', 'Rice, paddy')
        year  = int(d.get('year', 2024))
        rain  = float(d.get('rainfall', 1000))
        pest  = float(d.get('pesticides', 100))
        temp  = float(d.get('avg_temp', 24))

        if 'lat' in d and 'lon' in d:
            climate = fetch_climate(d['lat'], d['lon'])
            if climate is not None and climate.shape[0] >= 30:
                recent = climate[-7:]
                temp = float(np.mean(recent[:, 0]))
                hum  = float(np.mean(recent[:, 1]))
                prec = float(np.mean(recent[:, 2]))

        if area in area_encoder.classes_:
            area_code = area_encoder.transform([area])[0]
        else:
            area_code = 0
        if item in item_encoder.classes_:
            item_code = item_encoder.transform([item])[0]
        else:
            item_code = 0

        log_pest = np.log1p(pest)
        num_feat = num_scaler.transform([[year, rain, log_pest, temp]])
        pred_s = yield_model.predict(
            [np.array([area_code]), np.array([item_code]), num_feat],
            verbose=0
        )[0][0]
        t_ha = float(y_scaler.inverse_transform([[pred_s]])[0][0])

        field_area = float(d.get('area_ha', 1.0))
        return jsonify({
            'yield_per_ha': round(t_ha, 3),
            'total_yield':  round(t_ha * field_area, 3),
            'unit':         'tonnes/hectare'
        })
    except Exception as e:
        log.exception('predict-yield failed')
        return _err(str(e))


@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    try:
        if 'image' not in request.files:
            return _err('No image file provided')
        file = request.files['image']
        img  = Image.open(io.BytesIO(file.read())).convert('RGB')
        img  = img.resize((224, 224))
        arr  = np.array(img, dtype=np.float32) / 255.0
        arr  = np.expand_dims(arr, axis=0)
        proba = disease_model.predict(arr, verbose=0)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        return jsonify({
            'disease_class': class_names[top3_idx[0]],
            'confidence':    round(float(proba[top3_idx[0]] * 100), 2),
            'top3_classes':  [{'class': class_names[i],
                               'confidence': round(float(proba[i]*100), 2)}
                              for i in top3_idx]
        })
    except Exception as e:
        log.exception('predict-disease failed')
        return _err(str(e))


@app.route('/predict-yield-raw', methods=['POST'])
def predict_yield_raw():
    try:
        d = request.get_json()
        if not d:
            return _err('Request body must be JSON')
        area_code = int(d.get('area_code', 0))
        item_code = int(d.get('item_code', 0))
        year = int(d.get('year', 2024))
        rain = float(d.get('rainfall', 1000))
        pest = float(d.get('pesticides', 100))
        temp = float(d.get('avg_temp', 24))
        field_area = float(d.get('area_ha', 1.0))

        if 'lat' in d and 'lon' in d:
            climate = fetch_climate(d['lat'], d['lon'])
            if climate is not None and climate.shape[0] >= 30:
                recent = climate[-7:]
                temp = float(np.mean(recent[:, 0]))
                prec = float(np.mean(recent[:, 2]))
                rain = prec

        log_pest = np.log1p(pest)
        num_feat = num_scaler.transform([[year, rain, log_pest, temp]])
        pred_s = yield_model.predict(
            [np.array([area_code]), np.array([item_code]), num_feat],
            verbose=0
        )[0][0]
        t_ha = float(y_scaler.inverse_transform([[pred_s]])[0][0])
        return jsonify({
            'yield_per_ha': round(t_ha, 3),
            'total_yield': round(t_ha * float(field_area), 3),
            'unit': 'tonnes/hectare'
        })
    except Exception as e:
        log.exception('predict-yield-raw failed')
        return _err(str(e))


@app.route('/encoders', methods=['GET'])
def get_encoders():
    return jsonify({
        'areas': list(area_encoder.classes_),
        'items': list(item_encoder.classes_),
        'area_map': {a: int(i) for i, a in enumerate(area_encoder.classes_)},
        'item_map': {it: int(i) for i, it in enumerate(item_encoder.classes_)},
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)