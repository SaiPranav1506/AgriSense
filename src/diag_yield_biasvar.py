import sys; sys.path.insert(0, 'src')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np, pickle, json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization,
    Embedding, Flatten, Concatenate, Reshape)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# silence tf
tf.get_logger().setLevel('ERROR')

def build_model(n_areas, n_items):
    area_inp = Input(shape=(1,), name='area')
    item_inp = Input(shape=(1,), name='item')
    num_inp = Input(shape=(4,), name='numeric')
    area_emb = Embedding(n_areas, 16)(area_inp); area_emb = Flatten()(area_emb)
    item_emb = Embedding(n_items, 8)(item_inp); item_emb = Flatten()(item_emb)
    x = Concatenate()([area_emb, item_emb, num_inp])
    x = Dense(128, activation='relu')(x); x = BatchNormalization()(x); x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x); x = BatchNormalization()(x); x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x); x = Dropout(0.1)(x)
    out = Dense(1)(x)
    return Model([area_inp, item_inp, num_inp], out)

# ---------- DATA ----------
df = pd.read_csv('data/raw/yield_df.csv').dropna().reset_index(drop=True)
df['t_ha'] = df['hg/ha_yield'] / 10000
df['log_pest'] = np.log1p(df['pesticides_tonnes'])

area_enc = pickle.load(open('models/area_encoder.pkl','rb'))
item_enc = pickle.load(open('models/item_encoder.pkl','rb'))
df['area_code'] = area_enc.transform(df['Area'])
df['item_code'] = item_enc.transform(df['Item'])
n_areas = len(area_enc.classes_)
n_items = len(item_enc.classes_)

num_features = ['Year','average_rain_fall_mm_per_year','log_pest','avg_temp']
X_area = df['area_code'].values.astype(np.int32)
X_item = df['item_code'].values.astype(np.int32)
X_num = df[num_features].values.astype(np.float32)
y = df['t_ha'].values.astype(np.float32)

indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.176, random_state=42)

num_scaler = pickle.load(open('models/num_scaler.pkl','rb'))
y_scaler = pickle.load(open('models/y_scaler.pkl','rb'))

X_num_train = num_scaler.transform(X_num[train_idx])
X_num_val = num_scaler.transform(X_num[val_idx])
X_num_test = num_scaler.transform(X_num[test_idx])

def to_ha(pred_s):
    return y_scaler.inverse_transform(np.asarray(pred_s).reshape(-1,1)).ravel()

# ---------- PRODUCTION MODEL METRICS ----------
prod = load_model('models/cnn_lstm_yield.h5', compile=False)
yh_s = prod.predict([X_area[test_idx], X_item[test_idx], X_num_test], verbose=0).ravel()
yh = to_ha(yh_s)
yt = y[test_idx]

def metrics(yt_, yh_):
    r2 = r2_score(yt_, yh_)
    rmse = np.sqrt(mean_squared_error(yt_, yh_))
    mae = np.mean(np.abs(yt_-yh_))
    mape = np.mean(np.abs((yt_-yh_)/yt_))*100
    return r2, rmse, mae, mape

r2, rmse, mae, mape = metrics(yt, yh)
resid = yt - yh
resid_mean = float(np.mean(resid))
corr_abs = float(np.corrcoef(np.abs(resid), yt)[0,1])

# ---------- BOOTSTRAP BIAS/VARIANCE ----------
M = 8
n_test = len(test_idx)
predM = np.zeros((M, n_test))
np.random.seed(0)
for m in range(M):
    boot_idx = np.random.choice(train_idx, size=len(train_idx), replace=True)
    # only keep boot samples that have at least a couple per combo; fine as is
    Xn_b = num_scaler.transform(X_num[boot_idx])
    ys_b = y_scaler.transform(y[boot_idx].reshape(-1,1)).ravel()
    model = build_model(n_areas, n_items)
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
    cbs = [EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss')]
    model.fit([X_area[boot_idx], X_item[boot_idx], Xn_b], ys_b,
              validation_data=([X_area[val_idx], X_item[val_idx], X_num_val],
                               y_scaler.transform(y[val_idx].reshape(-1,1)).ravel()),
              epochs=20, batch_size=128, callbacks=cbs, verbose=0)
    predM[m] = to_ha(model.predict([X_area[test_idx], X_item[test_idx], X_num_test], verbose=0).ravel())
    del model

mean_pred = predM.mean(axis=0)
variance = float(predM.var(axis=0).mean())
bias_sq = float(((mean_pred - yt)**2).mean())
mse = float(((predM - yt[None,:])**2).mean())
ens_r2 = float(r2_score(yt, mean_pred))
bias_var = 'BIAS' if bias_sq >= variance else 'VARIANCE'

# ---------- RESIDUAL DIAGNOSTICS ----------
test_df = df.iloc[test_idx].copy()
test_df['resid'] = resid
test_df['abs_resid'] = np.abs(resid)

def top(dfg, col_label, n=5, neg=True):
    g = dfg.groupby(col_label)['resid'].mean()
    return g.sort_values() if neg else g.sort_values(ascending=False)

top_over_area = top(test_df,'Area',neg=True).head(5)
top_under_area = top(test_df,'Area',neg=False).head(5)
top_over_item = top(test_df,'Item',neg=True).head(5)
top_under_item = top(test_df,'Item',neg=False).head(5)

mape_item = test_df.groupby('Item').apply(
    lambda g: np.mean(np.abs(g['resid']/g['t_ha']))*100, include_groups=False)
worst_mape = mape_item.sort_values(ascending=False).head(5)

def fmt(g, pred=False):
    out = {}
    for k, v in g.items():
        if pred:
            un = item_enc.inverse_transform([k])[0] if k in item_enc.classes_ else k
        out[k] = round(float(v),3)
    return out

out = {
 'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)), 'n_test': int(len(test_idx)),
 'areas': n_areas, 'items': n_items, 'M': M,
 'prod': {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
          'residuals_mean': resid_mean, 'corr_absresid_y': corr_abs},
 'bv': {'variance': variance, 'bias_sq': bias_sq, 'mse': mse,
        'ensemble_r2': ens_r2, 'bias_vs_variance': bias_var,
        'sum_bias_var': bias_sq+variance},
 'over_area': {k: round(float(v),3) for k,v in top_over_area.items()},
 'under_area': {k: round(float(v),3) for k,v in top_under_area.items()},
 'over_item': {k: round(float(v),3) for k,v in top_over_item.items()},
 'under_item': {k: round(float(v),3) for k,v in top_under_item.items()},
 'worst_mape_item': {k: round(float(v),3) for k,v in worst_mape.items()},
}
print(json.dumps(out, indent=2, default=str))
with open('outputs/diag_yield_result.json','w') as f:
    json.dump(out, f, indent=2, default=str)
print('saved outputs/diag_yield_result.json')