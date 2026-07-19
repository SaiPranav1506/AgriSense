import sys, os, io, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
from weather_client import fetch_climate

app = Flask(__name__)

crop_model    = pickle.load(open('models/xgb_crop.pkl',              'rb'))
yield_model   = load_model('models/cnn_lstm_yield.h5', compile=False)
disease_model = load_model('models/mobilenet_disease.h5', compile=False)
label_encoder = pickle.load(open('models/label_encoder.pkl',         'rb'))
scaler        = pickle.load(open('models/scaler.pkl',                'rb'))
y_scaler      = pickle.load(open('models/y_scaler.pkl',              'rb'))
class_names   = pickle.load(open('models/disease_class_names.pkl',   'rb'))

# Yield model additional artifacts
area_encoder  = pickle.load(open('models/area_encoder.pkl',          'rb'))
item_encoder  = pickle.load(open('models/item_encoder.pkl',          'rb'))
num_scaler    = pickle.load(open('models/num_scaler.pkl',            'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict-crop', methods=['POST'])
def predict_crop():
    try:
        d    = request.get_json()
        N    = float(d['N'])
        P    = float(d['P'])
        K    = float(d['K'])
        temp = float(d['temperature'])
        hum  = float(d['humidity'])
        ph   = float(d['ph'])
        rain = float(d['rainfall'])
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
        return jsonify({'error': str(e)}), 400


@app.route('/predict-yield', methods=['POST'])
def predict_yield():
    try:
        d     = request.get_json()
        area  = d['area']
        item  = d['crop']
        year  = int(d.get('year', 2024))
        rain  = float(d.get('rainfall', 1000))
        pest  = float(d.get('pesticides', 100))
        temp  = float(d.get('avg_temp', 24))

        # Use live weather if lat/lon provided
        if 'lat' in d and 'lon' in d:
            climate = fetch_climate(d['lat'], d['lon'])
            if climate is not None and climate.shape[0] >= 30:
                recent = climate[-7:]
                temp = float(np.mean(recent[:, 0]))
                hum  = float(np.mean(recent[:, 1]))
                prec = float(np.mean(recent[:, 2]))

        # Encode categoricals (unknown -> most common, code=0)
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
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400


@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    try:
        file  = request.files['image']
        img   = Image.open(io.BytesIO(file.read())).convert('RGB')
        img   = img.resize((224, 224))
        arr   = np.array(img, dtype=np.float32) / 255.0
        arr   = np.expand_dims(arr, axis=0)
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
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)