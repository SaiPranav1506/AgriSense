import sys; sys.path.insert(0, 'src')
import numpy as np, pickle, os, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization,
    Embedding, Flatten, Concatenate, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def build_model(n_areas, n_items):
    area_inp = Input(shape=(1,), name='area')
    item_inp = Input(shape=(1,), name='item')
    num_inp = Input(shape=(4,), name='numeric')

    area_emb = Embedding(n_areas, 16)(area_inp)
    area_emb = Flatten()(area_emb)
    item_emb = Embedding(n_items, 8)(item_inp)
    item_emb = Flatten()(item_emb)

    x = Concatenate()([area_emb, item_emb, num_inp])
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(1)(x)
    return Model([area_inp, item_inp, num_inp], out)


def train():
    t0 = time.time()
    print("=" * 60)
    print("YIELD v2 | log1p target + same DNN architecture")
    print("=" * 60, flush=True)

    df = pd.read_csv('data/raw/yield_df.csv').dropna()
    df['t_ha'] = df['hg/ha_yield'] / 10000
    df['log_pest'] = np.log1p(df['pesticides_tonnes'])
    print(f"Yield data: {df.shape}", flush=True)

    area_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df['area_code'] = area_enc.fit_transform(df['Area'])
    df['item_code'] = item_enc.fit_transform(df['Item'])
    n_areas = df['area_code'].nunique()
    n_items = df['item_code'].nunique()
    print(f"Areas: {n_areas}, Items: {n_items}", flush=True)

    pickle.dump(area_enc, open('models/area_encoder.pkl', 'wb'))
    pickle.dump(item_enc, open('models/item_encoder.pkl', 'wb'))

    num_features = ['Year', 'average_rain_fall_mm_per_year', 'log_pest', 'avg_temp']
    X_area = df['area_code'].values.astype(np.int32)
    X_item = df['item_code'].values.astype(np.int32)
    X_num = df[num_features].values.astype(np.float32)

    y_orig = df['t_ha'].values.astype(np.float32)
    y = np.log1p(y_orig)
    print(f"Target: log1p(yield), mean={y.mean():.3f}, std={y.std():.3f}", flush=True)

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.176, random_state=42)

    num_scaler = StandardScaler()
    X_num_train = num_scaler.fit_transform(X_num[train_idx])
    X_num_val = num_scaler.transform(X_num[val_idx])
    X_num_test = num_scaler.transform(X_num[test_idx])
    pickle.dump(num_scaler, open('models/num_scaler_v2.pkl', 'wb'))

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    model = build_model(n_areas, n_items)
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
    model.summary()

    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    callbacks = [
        EarlyStopping(patience=12, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint('models/cnn_lstm_yield_v2.h5', save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss')
    ]

    history = model.fit(
        [X_area[train_idx], X_item[train_idx], X_num_train], y_train,
        validation_data=([X_area[val_idx], X_item[val_idx], X_num_val], y_val),
        epochs=100, batch_size=64, callbacks=callbacks, verbose=1
    )

    y_pred_log = model.predict([X_area[test_idx], X_item[test_idx], X_num_test], verbose=0).ravel()
    y_pred = np.expm1(y_pred_log)
    y_true = y_orig[test_idx]

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n  ===== FINAL (test, n={len(y_true)}) =====", flush=True)
    print(f"  R2  : {r2:.4f}", flush=True)
    print(f"  RMSE: {rmse:.4f} t/ha", flush=True)
    print(f"  MAE : {mae:.4f} t/ha", flush=True)
    print(f"  MAPE: {mape:.2f}%", flush=True)

    val_loss = history.history['val_loss']
    print(f"\n  Best val_loss: {min(val_loss):.4f} at epoch {np.argmin(val_loss)+1}", flush=True)
    print(f"  Saved: models/cnn_lstm_yield_v2.h5", flush=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, alpha=0.3, color='#4CAF6A', s=8)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', lw=1)
    plt.xlabel('Actual (t/ha)')
    plt.ylabel('Predicted (t/ha)')
    plt.title(f'Yield v2 (log-target): Actual vs Predicted (R2={r2:.3f}, MAPE={mape:.1f}%)')
    plt.tight_layout()
    plt.savefig('outputs/yield_v2_plot.png', dpi=150)
    print(f"  Saved: outputs/yield_v2_plot.png ({time.time()-t0:.0f}s)", flush=True)

    print("=" * 60)


if __name__ == '__main__':
    train()