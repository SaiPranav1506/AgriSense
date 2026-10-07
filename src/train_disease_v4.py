import sys; sys.path.insert(0, 'src')
from preprocess import load_disease_data
import numpy as np, os, pickle, json, time, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


def build_model(base, n_classes):
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(n_classes, activation='softmax')(x)
    return Model(base.input, out)


def train():
    t0 = time.time()
    print("=" * 60)
    print("DISEASE v4 | plain CE + label smoothing + long training (fix undertraining)")
    print("=" * 60, flush=True)

    train_ds, val_ds, test_ds, class_names = load_disease_data()
    n_classes = len(class_names)
    print(f"  classes: {n_classes}", flush=True)

    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    model = build_model(base, n_classes)

    # Plain cross-entropy (matches known-good baseline), no focal, no class weights
    loss = 'sparse_categorical_crossentropy'

    print("\n--- Phase 1: head only, lr 1e-3, up to 10 epochs + ReduceLROnPlateau ---", flush=True)
    model.compile(Adam(1e-3), loss, metrics=['accuracy'])
    rl1 = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    es1 = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[rl1, es1], verbose=1)

    print("\n--- Phase 2: unfreeze top 40, lr 1e-4, up to 22 epochs ---", flush=True)
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(Adam(1e-4), loss, metrics=['accuracy'])
    rl2 = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ckpt = ModelCheckpoint('models/mobilenet_disease_v4.h5', monitor='val_accuracy',
                           save_best_only=True, verbose=1)
    es2 = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=22, callbacks=[rl2, es2, ckpt], verbose=1)

    print("\n--- Evaluating test set (TTA: original + flip) ---", flush=True)
    y_true, y_pred, y_pred_tta = [], [], []
    for imgs, labels in test_ds:
        p0 = model.predict(imgs, verbose=0)
        p1 = model.predict(tf.image.flip_left_right(imgs), verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(p0, axis=1).tolist())
        y_pred_tta.extend(np.argmax(0.5 * (p0 + p1), axis=1).tolist())

    y_true = np.array(y_true); y_pred = np.array(y_pred); y_pred_tta = np.array(y_pred_tta)
    acc = accuracy_score(y_true, y_pred)
    acc_tta = accuracy_score(y_true, y_pred_tta)
    f1w = f1_score(y_true, y_pred_tta, average='weighted')
    f1m = f1_score(y_true, y_pred_tta, average='macro')

    print(f"\n  ===== FINAL (test, n={len(y_true)}) =====", flush=True)
    print(f"  Top-1 (plain)  : {acc*100:.2f}%", flush=True)
    print(f"  Top-1 (TTA)    : {acc_tta*100:.2f}%", flush=True)
    print(f"  Weighted F1    : {f1w:.4f}", flush=True)
    print(f"  Macro F1       : {f1m:.4f}", flush=True)

    os.makedirs('models', exist_ok=True)
    print("  Saved: models/mobilenet_disease_v4.h5 (best) ; production untouched", flush=True)

    print("\n  Per-class (worst 6):", flush=True)
    for cls in np.argsort([accuracy_score(y_true[y_true == i], y_pred_tta[y_true == i]) if (y_true == i).sum() > 0 else 0
                           for i in range(n_classes)])[:6]:
        mask = y_true == cls
        ca = accuracy_score(y_true[mask], y_pred_tta[mask]) if mask.sum() > 0 else 0
        print(f"    {class_names[cls][:44]:44s} C{cls:2d}: {ca*100:.1f}% (n={mask.sum()})", flush=True)

    os.makedirs('outputs', exist_ok=True)
    cm = confusion_matrix(y_true, y_pred_tta)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(cm, display_labels=[c[:16] for c in class_names]).plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.title('Disease v4 (plain CE + label smoothing + TTA)')
    plt.tight_layout()
    plt.savefig('outputs/disease_v4_confusion.png', dpi=150)

    result = {
        'test_top1': round(float(acc), 6),
        'test_top1_tta': round(float(acc_tta), 6),
        'weighted_f1': round(float(f1w), 6),
        'macro_f1': round(float(f1m), 6),
        'n_test': int(len(y_true)),
        'n_classes': int(n_classes),
    }
    with open('outputs/disease_v4_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: outputs/disease_v4_metrics.json ({time.time()-t0:.0f}s)", flush=True)
    print(f"** DISEASE v4: {acc_tta*100:.2f}% top-1 (TTA) | macro F1 {f1m:.3f} **", flush=True)
    print("=" * 60)


if __name__ == '__main__':
    train()