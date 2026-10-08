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

_IMG = (128, 128)
_PHASE1_LR = 1e-3
_PHASE2_LR = 1e-5
_NCLS = None
_LS = 0.1


def smoothed_loss(y_true, y_pred):
    y_true_onehot = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), _NCLS)
    y_smooth = y_true_onehot * (1.0 - _LS) + _LS / tf.cast(_NCLS, tf.float32)
    return tf.keras.losses.categorical_crossentropy(y_smooth, y_pred)


def build_model(base, n_classes):
    global _NCLS
    _NCLS = n_classes
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    out = Dense(n_classes, activation='softmax')(x)
    return Model(base.input, out)


def train():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("DISEASE v5 | Conservative Fine-tuning", flush=True)
    print(f"  img={_IMG}, loss=label_smoothing(0.1)", flush=True)
    print(f"  Phase1: frozen, lr={_PHASE1_LR}, epochs=10", flush=True)
    print(f"  Phase2: unfreeze last 3, lr={_PHASE2_LR}, clipnorm=1.0, epochs=25", flush=True)
    print("=" * 60, flush=True)

    train_ds, val_ds, test_ds, class_names = load_disease_data(img_size=_IMG)
    n_classes = len(class_names)
    print(f"  classes: {n_classes}", flush=True)

    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*_IMG, 3))
    base.trainable = False
    model = build_model(base, n_classes)

    y_train_list = [y.numpy() for _, y in train_ds]
    y_train_all = np.concatenate(y_train_list)
    classes = np.arange(n_classes)
    cw = compute_class_weight('balanced', classes=classes, y=y_train_all)
    cw = np.sqrt(cw)
    cw_dict = {i: float(cw[i]) for i in range(n_classes)}
    print(f"  Class weights range: [{cw.min():.2f}, {cw.max():.2f}] (sqrt-softened)", flush=True)

    # ── Phase 1: frozen backbone ──
    print("\n--- Phase 1: frozen backbone, lr=1e-3 ---", flush=True)
    model.compile(optimizer=Adam(_PHASE1_LR), loss=smoothed_loss, metrics=['accuracy'])
    rl1 = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    es1 = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
    ckpt1 = ModelCheckpoint('models/mobilenet_disease_v5.h5', monitor='val_accuracy',
                            save_best_only=True, verbose=1)
    model.fit(train_ds, validation_data=val_ds, epochs=10,
              callbacks=[rl1, es1, ckpt1], verbose=1)

    # Save Phase 1 weights separately before unfreezing
    phase1_path = 'models/disease_v5_phase1.h5'
    model.save(phase1_path)
    print(f"\n  Phase 1 best model saved to {phase1_path}", flush=True)

    # ── Phase 2: conservative fine-tuning ──
    print("\n--- Phase 2: unfreeze last 3 layers, lr=1e-5, clipnorm=1.0 ---", flush=True)
    base.trainable = True
    n_total = len(base.layers)
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= n_total - 3)

    trainable = sum(1 for l in base.layers if l.trainable)
    print(f"  Unfrozen layers: {trainable} (last 3: {base.layers[-3].name}, {base.layers[-2].name}, {base.layers[-1].name})", flush=True)

    model.compile(
        optimizer=Adam(_PHASE2_LR, clipnorm=1.0),
        loss=smoothed_loss,
        metrics=['accuracy']
    )

    snap_dir = 'models/snapshots'
    os.makedirs(snap_dir, exist_ok=True)
    for f in os.listdir(snap_dir):
        try:
            os.remove(os.path.join(snap_dir, f))
        except OSError:
            pass

    class SnapCallback(tf.keras.callbacks.Callback):
        def __init__(self, save_dir, epochs_to_save):
            super().__init__()
            self.save_dir = save_dir
            self.epochs = sorted(set(epochs_to_save))
            self.idx = 0
        def on_epoch_end(self, epoch, logs=None):
            ep = epoch + 1
            while self.idx < len(self.epochs) and ep >= self.epochs[self.idx]:
                path = os.path.join(self.save_dir, f'snap_{self.idx}.h5')
                self.model.save(path)
                print(f"  [snapshot] saved snap_{self.idx}.h5 at epoch {ep}", flush=True)
                self.idx += 1

    rl2 = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ckpt2 = ModelCheckpoint('models/mobilenet_disease_v5.h5', monitor='val_accuracy',
                            save_best_only=True, verbose=1)
    es2 = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
    snap_cb = SnapCallback(snap_dir, [8, 16, 25])

    model.fit(train_ds, validation_data=val_ds, epochs=25,
              callbacks=[rl2, es2, ckpt2, snap_cb],
              class_weight=cw_dict, verbose=1)

    # ── Evaluate ──
    print("\n--- Evaluating test set (TTA: 5 views) ---", flush=True)
    best = tf.keras.models.load_model('models/mobilenet_disease_v5.h5', compile=False)
    snaps = sorted([os.path.join(snap_dir, f) for f in os.listdir(snap_dir) if f.endswith('.h5')])
    snap_models = [tf.keras.models.load_model(s, compile=False) for s in snaps]
    print(f"  Loaded {len(snap_models)} snapshot models for ensemble", flush=True)

    y_true, y_pred, y_pred_tta = [], [], []
    for imgs, labels in test_ds:
        preds = [m.predict(imgs, verbose=0) for m in ([best] + snap_models)]
        p0 = preds[0]
        p_ens = np.mean(preds, axis=0)
        p1 = best.predict(tf.image.flip_left_right(imgs), verbose=0)
        p2 = best.predict(tf.image.rot90(imgs, k=1), verbose=0)
        p3 = best.predict(tf.image.rot90(imgs, k=3), verbose=0)
        p4 = best.predict(tf.image.random_brightness(imgs, 0.15), verbose=0)
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
    print(f"  Top-1 (plain)       : {acc*100:.2f}%", flush=True)
    print(f"  Top-1 (TTA+5)       : {acc_tta*100:.2f}%", flush=True)
    print(f"  Weighted F1         : {f1w:.4f}", flush=True)
    print(f"  Macro F1            : {f1m:.4f}", flush=True)

    print("\n  Per-class (worst 8):", flush=True)
    per_cls = [accuracy_score(y_true[y_true == i], y_pred_tta[y_true == i])
               if (y_true == i).sum() > 0 else 0 for i in range(n_classes)]
    for cls in np.argsort(per_cls)[:8]:
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
    ConfusionMatrixDisplay(cm, display_labels=[c[:16] for c in class_names]).plot(
        ax=ax, xticks_rotation=90, colorbar=False)
    plt.title('Disease v5 (Conservative Fine-tuning + TTA+5 + Snapshot Ensemble)')
    plt.tight_layout()
    plt.savefig('outputs/disease_v5_confusion.png', dpi=150)

    result = {
        'test_top1': round(float(acc), 6),
        'test_top1_tta': round(float(acc_tta), 6),
        'weighted_f1': round(float(f1w), 6),
        'macro_f1': round(float(f1m), 6),
        'n_test': int(len(y_true)),
        'n_classes': int(n_classes),
        'n_snapshots': len(snaps),
        'label_smoothing': 0.1,
        'tta_views': 5,
        'phase1_lr': _PHASE1_LR,
        'phase2_lr': _PHASE2_LR,
        'unfrozen_layers': 3,
        'clipnorm': 1.0,
        'class_weight_method': 'sqrt-softened',
    }
    with open('outputs/disease_v5_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: outputs/disease_v5_metrics.json ({time.time()-t0:.0f}s)", flush=True)
    print(f"** DISEASE v5: {acc_tta*100:.2f}% top-1 (TTA+5) | macro F1 {f1m:.3f} **", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    train()