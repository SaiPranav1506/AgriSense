import sys; sys.path.insert(0, 'src')
import numpy as np, pickle, os
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
    df = pd.read_csv('data/raw/yield_df.csv').dropna()
    df['t_ha'] = df['hg/ha_yield'] / 10000
    df['log_pest'] = np.log1p(df['pesticides_tonnes'])
    print(f"Yield data: {df.shape}")

    area_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df['area_code'] = area_enc.fit_transform(df['Area'])
    df['item_code'] = item_enc.fit_transform(df['Item'])
    n_areas = df['area_code'].nunique()
    n_items = df['item_code'].nunique()
    print(f"Areas: {n_areas}, Items: {n_items}")
    # Save encoders for inference
    pickle.dump(area_enc, open('models/area_encoder.pkl', 'wb'))
    pickle.dump(item_enc, open('models/item_encoder.pkl', 'wb'))

    num_features = ['Year', 'average_rain_fall_mm_per_year', 'log_pest', 'avg_temp']
    X_area = df['area_code'].values.astype(np.int32)
    X_item = df['item_code'].values.astype(np.int32)
    X_num = df[num_features].values.astype(np.float32)
    y = df['t_ha'].values.astype(np.float32)

    # Train/test split
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.176, random_state=42)

    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y[train_idx].reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y[val_idx].reshape(-1, 1)).ravel()
    pickle.dump(y_scaler, open('models/y_scaler.pkl', 'wb'))

    num_scaler = StandardScaler()
    X_num_train = num_scaler.fit_transform(X_num[train_idx])
    X_num_val = num_scaler.transform(X_num[val_idx])
    X_num_test = num_scaler.transform(X_num[test_idx])
    pickle.dump(num_scaler, open('models/num_scaler.pkl', 'wb'))

    model = build_model(n_areas, n_items)
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
    model.summary()

    os.makedirs('outputs', exist_ok=True)
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint('models/cnn_lstm_yield.h5', save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss')
    ]

    history = model.fit(
        [X_area[train_idx], X_item[train_idx], X_num_train],
        y_train_s,
        validation_data=([X_area[val_idx], X_item[val_idx], X_num_val], y_val_s),
        epochs=100, batch_size=64, callbacks=callbacks, verbose=1
    )

    y_pred_s = model.predict([X_area[test_idx], X_item[test_idx], X_num_test]).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_test = y[test_idx]

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\nR2  : {r2:.4f}")
    print(f"RMSE: {rmse:.4f} (t/ha)")
    print(f"MAE : {mae:.4f} (t/ha)")

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.3, color='#4CAF6A', s=8)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', lw=1)
    plt.xlabel('Actual (t/ha)')
    plt.ylabel('Predicted (t/ha)')
    plt.title(f'DNN: Actual vs Predicted Yield (R²={r2:.3f})')
    plt.tight_layout()
    plt.savefig('outputs/yield_plot.png', dpi=150)

    val_loss = history.history['val_loss']
    print(f"Best val_loss: {min(val_loss):.4f} at epoch {np.argmin(val_loss)+1}")
    print("Saved: models/cnn_lstm_yield.h5, outputs/yield_plot.png")


if __name__ == '__main__':
    train()