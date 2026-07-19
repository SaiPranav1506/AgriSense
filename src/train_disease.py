import sys; sys.path.insert(0, 'src')
from preprocess import load_disease_data
import numpy as np, pickle, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def train():
    train_ds, val_ds, test_ds, class_names = load_disease_data()
    n_classes = len(class_names)

    base = MobileNetV2(weights='imagenet', include_top=False,
                       input_shape=(224, 224, 3))
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(base.input, out)

    callbacks = [
        EarlyStopping(patience=4, restore_best_weights=True),
        ModelCheckpoint('models/mobilenet_disease.h5', save_best_only=True)
    ]

    print(f"\n--- Phase 1: Training head only ({n_classes} classes) ---")
    model.compile(Adam(1e-3), 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds,
              epochs=5, callbacks=callbacks)

    print("\n--- Phase 2: Fine-tuning top 30 base layers ---")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(Adam(1e-5), 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds,
              epochs=10, callbacks=callbacks)

    loss, acc = model.evaluate(test_ds)
    print(f"\nTest accuracy: {acc*100:.2f}%")

    os.makedirs('outputs', exist_ok=True)

    y_true, y_pred = [], []
    for imgs, labels in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    # Top 10 most common classes in test set
    top10 = np.argsort(np.bincount(y_true))[-10:]
    mask = [i for i, v in enumerate(y_true) if v in top10]
    yt10 = [y_true[i] for i in mask]
    yp10 = [y_pred[i] for i in mask]
    cm = confusion_matrix(yt10, yp10, labels=top10)

    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(
        cm, display_labels=[class_names[i] for i in top10]
    ).plot(ax=ax, xticks_rotation=45)
    plt.title('MobileNetV2: Confusion Matrix (Top 10 Classes)')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    print("Saved: outputs/confusion_matrix.png")


if __name__ == '__main__':
    train()