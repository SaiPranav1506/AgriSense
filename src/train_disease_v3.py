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
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class SparseFocalLoss(tf.keras.losses.Loss):
    """Sparse focal loss with per-class weights."""
    def __init__(self, gamma=2.0, class_weights=None, name='sparse_focal'):
        super().__init__(name=name)
        self.gamma = gamma
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-9, 1.0)
        yt = tf.cast(y_true, tf.int32)
        onehot = tf.one_hot(yt, tf.shape(y_pred)[-1])
        pt = tf.reduce_sum(onehot * y_pred, axis=-1)
        loss = -tf.pow(1.0 - pt, self.gamma) * tf.math.log(pt)
        if self.class_weights is not None:
            loss = loss * tf.gather(self.class_weights, yt)
        return loss


def class_weights(class_names, data_dir='data/raw/PlantVillage'):
    counts = []
    for c in class_names:
        d = os.path.join(data_dir, c)
        n = len([f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        counts.append(max(n, 1))
    counts = np.array(counts, dtype=np.float32)
    bal = counts.sum() / (len(counts) * counts)
    return np.power(bal, 0.5), counts  # sqrt-softened to avoid minority-collapse


def build_model(base, n_classes):
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(n_classes, activation='softmax')(x)
    return Model(base.input, out)


def train():
    t0 = time.time()
    print("=" * 60)
    print("DISEASE v3 | fixed LR schedule + sqrt-softened focal loss")
    print("=" * 60, flush=True)

    train_ds, val_ds, test_ds, class_names = load_disease_data()
    n_classes = len(class_names)
    cw, counts = class_weights(class_names)
    cw_t = tf.constant(cw, dtype=tf.float32)
    steps_per_epoch = max(len(train_ds), 1)
    print(f"  classes: {n_classes}, steps/epoch: {steps_per_epoch}", flush=True)
    print(f"  image counts: {counts.tolist()}", flush=True)

    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    model = build_model(base, n_classes)
    focal = SparseFocalLoss(gamma=2.0, class_weights=cw_t)

    print("\n--- Phase 1: head only, lr cosine 1e-3 over entire phase ---", flush=True)
    model.compile(Adam(CosineDecay(1e-3, steps_per_epoch * 8, alpha=0.01)), loss=focal, metrics=['accuracy'])
    es1 = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=[es1], verbose=1)

    print("\n--- Phase 2: unfreeze top 60 layers, lr cosine 1e-4 ---", flush=True)
    base.trainable = True
    for layer in base.layers[:-60]:
        layer.trainable = False
    model.compile(Adam(CosineDecay(1e-4, steps_per_epoch * 14, alpha=0.01)), loss=focal, metrics=['accuracy'])
    ckpt = ModelCheckpoint('models/mobilenet_disease_v3.h5', monitor='val_accuracy',
                           save_best_only=True, verbose=1)
    es2 = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=14, callbacks=[es2, ckpt], verbose=1)

    print("\n--- Evaluating test set (with TTA: original + flip) ---", flush=True)
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
    try:
        model.save('models/mobilenet_disease_v3_final.h5')
    except Exception as e:
        print(f"  (extra final save skipped: {e})", flush=True)
    print("  Saved: models/mobilenet_disease_v3.h5 (best) ; production untouched", flush=True)

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
    plt.title('Disease v3 (focal + TTA)')
    plt.tight_layout()
    plt.savefig('outputs/disease_v3_confusion.png', dpi=150)

    result = {
        'test_top1': round(float(acc), 6),
        'test_top1_tta': round(float(acc_tta), 6),
        'weighted_f1': round(float(f1w), 6),
        'macro_f1': round(float(f1m), 6),
        'n_test': int(len(y_true)),
        'n_classes': int(n_classes),
        'class_counts': [int(c) for c in counts],
    }
    with open('outputs/disease_v3_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: outputs/disease_v3_metrics.json ({time.time()-t0:.0f}s)", flush=True)
    print(f"** DISEASE v3: {acc_tta*100:.2f}% top-1 (TTA) | macro F1 {f1m:.3f} **", flush=True)
    print("=" * 60)


if __name__ == '__main__':
    train()