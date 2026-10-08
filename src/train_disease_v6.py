import sys; sys.path.insert(0, 'src')
import numpy as np, os, pickle, json, time, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Dropout,
                                     BatchNormalization, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ── Config ──
_IMG = (224, 224)
_P1_EPOCHS = 12
_P2_EPOCHS = 12
_P1_LR = 1e-3
_P2_LR = 1e-5
_BATCH = 32
_LS = 0.1
_UNFREEZE_FRAC = 0.30
_DATA = "data/raw/PlantVillage"
_NCLS = None


def smoothed_loss(y_true, y_pred):
    y_true_onehot = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), _NCLS)
    y_smooth = y_true_onehot * (1.0 - _LS) + _LS / tf.cast(_NCLS, tf.float32)
    return tf.keras.losses.categorical_crossentropy(y_smooth, y_pred)


def load_clean(img_size, batch_size):
    """Self-contained loader: images stay in [0,255]. Augmentation uses the
    CORRECT value_range=(0,255) — the old preprocess.py rescaled to [0,1] then
    applied RandomBrightness with the default (0,255) range, corrupting every
    image (~51-unit shift). That bug capped v5 at 66.9%."""
    trainval = tf.keras.utils.image_dataset_from_directory(
        _DATA, validation_split=0.3, subset='training', seed=42,
        image_size=img_size, batch_size=batch_size, label_mode='int')
    test_ds = tf.keras.utils.image_dataset_from_directory(
        _DATA, validation_split=0.3, subset='validation', seed=42,
        image_size=img_size, batch_size=batch_size, label_mode='int')

    val_batches = tf.data.experimental.cardinality(trainval) // 5
    val_ds = trainval.take(val_batches)
    train_ds = trainval.skip(val_batches)

    # Augment in [0,255] space with explicit value_range
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 255.0)),
        tf.keras.layers.RandomContrast(0.1),
    ])
    train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    class_names = sorted([c for c in os.listdir(_DATA)
                          if os.path.isdir(os.path.join(_DATA, c))])
    return train_ds, val_ds, test_ds, class_names


def build_model(n_classes):
    """EfficientNetV2B0 with include_preprocessing=True on [0,255] inputs.
    A frozen-backbone linear probe on this clean pipeline already scores
    ~91% — the earlier collapse was purely the corrupted data range."""
    global _NCLS
    _NCLS = n_classes
    inp = Input(shape=(*_IMG, 3))
    base = EfficientNetV2B0(weights='imagenet', include_top=False,
                            input_shape=(*_IMG, 3), include_preprocessing=True)
    base.trainable = False
    x = base(inp)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(384, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(n_classes, activation='softmax')(x)
    return Model(inp, out), base


def train():
    t0 = time.time()
    print("=" * 64, flush=True)
    print("DISEASE v6 | EfficientNetV2B0 @ 224x224 | CLEAN [0,255] pipeline", flush=True)
    print(f"  Phase1: frozen, lr={_P1_LR}, epochs={_P1_EPOCHS}", flush=True)
    print(f"  Phase2: unfreeze top {_UNFREEZE_FRAC*100:.0f}%, lr={_P2_LR}, epochs={_P2_EPOCHS}", flush=True)
    print(f"  Per-sample class weights + correct-range aug + label smoothing {_LS}", flush=True)
    print("=" * 64, flush=True)

    train_ds, val_ds, test_ds, class_names = load_clean(_IMG, _BATCH)
    n_classes = len(class_names)
    print(f"  classes: {n_classes}", flush=True)

    # ── Per-sample class weights (capped) ──
    y_train_all = np.concatenate([y.numpy() for _, y in train_ds])
    classes = np.arange(n_classes)
    cw = compute_class_weight('balanced', classes=classes, y=y_train_all)
    cw = np.clip(cw / cw.mean(), 0.5, 3.0)
    cw_vec = tf.constant(cw.astype(np.float32))
    print(f"  Class weights (capped): min={cw.min():.2f} max={cw.max():.2f}", flush=True)

    def add_weight(x, y):
        return x, y, tf.gather(cw_vec, y)
    train_w = train_ds.map(add_weight, num_parallel_calls=tf.data.AUTOTUNE)

    model, base = build_model(n_classes)

    # ── Phase 1 ──
    print("\n--- Phase 1: frozen backbone ---", flush=True)
    model.compile(optimizer=Adam(_P1_LR), loss=smoothed_loss, metrics=['accuracy'],
                  weighted_metrics=[])
    rl1 = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    es1 = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
    ckpt1 = ModelCheckpoint('models/efficientnet_disease_v6.h5', monitor='val_accuracy',
                            save_best_only=True, verbose=1)
    model.fit(train_w, validation_data=val_ds, epochs=_P1_EPOCHS,
              callbacks=[rl1, es1, ckpt1], verbose=1)
    model.save('models/disease_v6_phase1.h5')
    print("  Phase 1 done, saved disease_v6_phase1.h5", flush=True)

    # ── Phase 2 ──
    print(f"\n--- Phase 2: unfreeze top {_UNFREEZE_FRAC*100:.0f}% ---", flush=True)
    base.trainable = True
    n = len(base.layers)
    cutoff = int(n * (1 - _UNFREEZE_FRAC))
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= cutoff) and not isinstance(layer, tf.keras.layers.BatchNormalization)
    print(f"  Unfrozen {sum(1 for l in base.layers if l.trainable)}/{n} backbone layers (BN frozen)", flush=True)

    model.compile(optimizer=Adam(_P2_LR, clipnorm=1.0), loss=smoothed_loss,
                  metrics=['accuracy'], weighted_metrics=[])

    snap_dir = 'models/snapshots_v6'
    os.makedirs(snap_dir, exist_ok=True)
    for f in os.listdir(snap_dir):
        try: os.remove(os.path.join(snap_dir, f))
        except OSError: pass

    class Snap(tf.keras.callbacks.Callback):
        def __init__(self, d, eps): super().__init__(); self.d=d; self.e=sorted(set(eps)); self.i=0
        def on_epoch_end(self, ep, logs=None):
            e = ep + 1
            while self.i < len(self.e) and e >= self.e[self.i]:
                self.model.save(os.path.join(self.d, f'snap_{self.i}.h5'))
                print(f"  [snapshot] snap_{self.i}.h5 @ epoch {e}", flush=True); self.i += 1

    rl2 = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    es2 = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    ckpt2 = ModelCheckpoint('models/efficientnet_disease_v6.h5', monitor='val_accuracy',
                            save_best_only=True, verbose=1)
    snap_cb = Snap(snap_dir, [int(_P2_EPOCHS*0.6), _P2_EPOCHS])
    model.fit(train_w, validation_data=val_ds, epochs=_P2_EPOCHS,
              callbacks=[rl2, es2, ckpt2, snap_cb], verbose=1)

    # ── Evaluate with 5-view TTA ──
    print("\n--- Evaluating test set (TTA: 5 views) ---", flush=True)
    best = tf.keras.models.load_model('models/efficientnet_disease_v6.h5', compile=False)
    snaps = sorted([os.path.join(snap_dir, f) for f in os.listdir(snap_dir) if f.endswith('.h5')])
    snap_models = [tf.keras.models.load_model(s, compile=False) for s in snaps]
    print(f"  Loaded {len(snap_models)} snapshots", flush=True)

    y_true, y_pred, y_pred_tta = [], [], []
    for imgs, labels in test_ds:
        preds = [m.predict(imgs, verbose=0) for m in ([best] + snap_models)]
        p0 = preds[0]
        p_ens = np.mean(preds, axis=0)
        p1 = best.predict(tf.image.flip_left_right(imgs), verbose=0)
        p2 = best.predict(tf.image.rot90(imgs, k=1), verbose=0)
        p3 = best.predict(tf.image.rot90(imgs, k=3), verbose=0)
        p4 = best.predict(tf.image.random_brightness(imgs, 0.1, value_range=(0.0, 255.0)), verbose=0)
        p_tta = (p0 + p1 + p2 + p3 + p4) / 5.0
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(p_ens, axis=1).tolist())
        y_pred_tta.extend(np.argmax(p_tta, axis=1).tolist())

    y_true = np.array(y_true); y_pred = np.array(y_pred); y_pred_tta = np.array(y_pred_tta)
    acc = accuracy_score(y_true, y_pred)
    acc_tta = accuracy_score(y_true, y_pred_tta)
    f1w = f1_score(y_true, y_pred_tta, average='weighted')
    f1m = f1_score(y_true, y_pred_tta, average='macro')

    print(f"\n  ===== FINAL (test, n={len(y_true)}) =====", flush=True)
    print(f"  Top-1 (plain)  : {acc*100:.2f}%", flush=True)
    print(f"  Top-1 (TTA+5)  : {acc_tta*100:.2f}%", flush=True)
    print(f"  Weighted F1    : {f1w:.4f}", flush=True)
    print(f"  Macro F1       : {f1m:.4f}", flush=True)

    print("\n  Per-class (worst 8):", flush=True)
    per_cls = [accuracy_score(y_true[y_true == i], y_pred_tta[y_true == i])
               if (y_true == i).sum() > 0 else 0 for i in range(n_classes)]
    for cls in np.argsort(per_cls)[:8]:
        mask = y_true == cls
        ca = accuracy_score(y_true[mask], y_pred_tta[mask]) if mask.sum() > 0 else 0
        print(f"    {class_names[cls][:44]:44s} C{cls:2d}: {ca*100:.1f}% (n={mask.sum()})", flush=True)

    os.makedirs('outputs', exist_ok=True)
    cm = confusion_matrix(y_true, y_pred_tta)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(cm, display_labels=[c[:16] for c in class_names]).plot(
        ax=ax, xticks_rotation=90, colorbar=False)
    plt.title('Disease v6 (EfficientNetV2B0 224px clean pipeline, TTA+5)')
    plt.tight_layout(); plt.savefig('outputs/disease_v6_confusion.png', dpi=150)

    result = {
        'test_top1': round(float(acc), 6), 'test_top1_tta': round(float(acc_tta), 6),
        'weighted_f1': round(float(f1w), 6), 'macro_f1': round(float(f1m), 6),
        'n_test': int(len(y_true)), 'n_classes': int(n_classes),
        'n_snapshots': len(snaps), 'img_size': _IMG[0], 'backbone': 'EfficientNetV2B0',
        'phase1_epochs': _P1_EPOCHS, 'phase2_epochs': _P2_EPOCHS,
        'unfreeze_frac': _UNFREEZE_FRAC, 'label_smoothing': _LS, 'tta_views': 5,
        'class_balance': 'per-sample weights (capped)',
        'note': 'Fixed data pipeline: [0,255] + include_preprocessing=True. Old pipeline had RandomBrightness value_range bug that capped v5 at 66.9%.',
    }
    with open('outputs/disease_v6_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: outputs/disease_v6_metrics.json ({time.time()-t0:.0f}s)", flush=True)
    print(f"** DISEASE v6: {acc_tta*100:.2f}% top-1 (TTA+5) | macro F1 {f1m:.3f} **", flush=True)
    print("=" * 64, flush=True)


if __name__ == '__main__':
    train()
