import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf


def load_crop_data(path="data/raw/Crop_recommendation_Extended.csv"):
    df = pd.read_csv(path)
    assert df.isnull().sum().sum() == 0, "Null values found in crop data"
    print(f"Crop data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    os.makedirs('models', exist_ok=True)
    pickle.dump(le, open('models/label_encoder.pkl', 'wb'))

    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[feature_cols].values).astype(np.float32)
    pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

    y = df['label'].values.astype(np.int32)
    classes = len(le.classes_)
    print(f"Crop data loaded. X={X.shape}, y={y.shape}, classes={classes}")
    return X, y, feature_cols


def build_yield_sequences(path="data/raw/yield_df.csv", window=30, save=True):
    df = pd.read_csv(path).dropna()
    print(f"Yield data rows after dropna: {len(df)}")

    df['t_ha'] = df['hg/ha_yield'] / 10000
    features = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Year']
    df['Year'] = df['Year'].astype(int)

    groups = df.groupby(['Item', 'Area'])
    X_list, y_list = [], []
    for name, group in groups:
        group = group.sort_values('Year')
        vals = group[features].values.astype(np.float32)
        targets = group['t_ha'].values.astype(np.float32)
        n = len(group)
        if n >= window + 1:
            for i in range(n - window):
                X_list.append(vals[i:i + window])
                y_list.append(targets[i + window])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"Yield sequences built. X={X.shape}, y={y.shape}")

    if save:
        os.makedirs('data/processed', exist_ok=True)
        np.save('data/processed/yield_sequences.npy', X)
        np.save('data/processed/yield_labels.npy', y)
    return X, y


def load_disease_data(path="data/raw/PlantVillage", img_size=(224, 224), batch_size=32):
    trainval_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.3,
        subset='training',
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int'
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.3,
        subset='validation',
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int'
    )

    val_batches = tf.data.experimental.cardinality(trainval_ds) // 5
    val_ds = trainval_ds.take(val_batches)
    train_ds = trainval_ds.skip(val_batches)

    rescale = tf.keras.layers.Rescaling(1. / 255)
    train_ds = train_ds.map(lambda x, y: (rescale(x), y))
    val_ds = val_ds.map(lambda x, y: (rescale(x), y))
    test_ds = test_ds.map(lambda x, y: (rescale(x), y))

    aug_layers = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomBrightness(0.2),
    ])
    train_ds = train_ds.map(lambda x, y: (aug_layers(x, training=True), y))

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    class_names = sorted(os.listdir(path))
    class_names = [c for c in class_names if os.path.isdir(os.path.join(path, c))]
    os.makedirs('models', exist_ok=True)
    pickle.dump(class_names, open('models/disease_class_names.pkl', 'wb'))

    print(f"Disease class names ({len(class_names)}):")
    for i, c in enumerate(class_names):
        print(f"  {i+1:2d}. {c}")

    return train_ds, val_ds, test_ds, class_names


if __name__ == '__main__':
    X, y, feats = load_crop_data()
    X_y, y_y = build_yield_sequences()
    print("Preprocessing complete.")