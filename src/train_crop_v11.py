import sys; sys.path.insert(0, 'src')
from crop_features import compute_legit_features, LEGIT_FEATURES
import numpy as np, pickle, os, json, time, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

N_CLASSES = 46

# Feature-set candidates, all label-free. base15 = the original leak-free subset.
BASE15 = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
          'N_pct', 'P_pct', 'K_pct', 'N_P_ratio', 'N_K_ratio', 'P_K_ratio',
          'hum_temp', 'rain_ph']
FULL22 = LEGIT_FEATURES  # all 22


def make_model():
    return XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.75, colsample_bytree=0.7,
        min_child_weight=8, gamma=0.2,
        reg_alpha=0.5, reg_lambda=6.0, max_delta_step=5,
        random_state=42, tree_method='hist', verbosity=0, n_jobs=4)


def softened_weights(y):
    cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
    return dict(zip(np.unique(y), np.power(cw, 0.5)))


def cv_score(X, y, sw_map):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(y), dtype=int)
    folds = []
    for tr, va in skf.split(X, y):
        w = np.array([sw_map[c] for c in y[tr]])
        m = make_model()
        m.fit(X[tr], y[tr], sample_weight=w, verbose=False)
        preds[va] = m.predict(X[va])
        folds.append(accuracy_score(y[va], preds[va]))
    return accuracy_score(y, preds), folds


def train():
    t0 = time.time()
    print("=" * 62)
    print("CROP MODEL v11 | LEAK-FREE retrain (no label-derived features)")
    print("=" * 62, flush=True)

    import pandas as pd
    df = pd.read_csv('data/raw/Crop_recommendation_Extended.csv')
    le = pickle.load(open('models/label_encoder.pkl', 'rb'))
    y = le.transform(df['label'].values).astype(np.int32)

    # All 22 leak-free features (fixed order = LEGIT_FEATURES)
    X_all = compute_legit_features(df['N'], df['P'], df['K'],
                                   df['temperature'], df['humidity'],
                                   df['ph'], df['rainfall'])
    print(f"  X={X_all.shape}, {len(np.unique(y))} crops, {len(LEGIT_FEATURES)} leak-free features", flush=True)

    sw_map = softened_weights(y)

    # ── Compare feature sets by CV ──
    print("\n[1/3] Feature-set selection (5-fold CV) ...", flush=True)
    results = {}
    for tag, cols in [('base15', BASE15), ('full22', FULL22)]:
        idx = [LEGIT_FEATURES.index(f) for f in cols]
        sc = StandardScaler().fit_transform(X_all[:, idx])
        acc, folds = cv_score(sc, y, sw_map)
        results[tag] = (acc, cols)
        print(f"    {tag:8s} ({len(cols)} feats): CV top-1 = {acc*100:.2f}%  "
              f"folds={[round(f*100,1) for f in folds]}", flush=True)

    best_tag = max(results, key=lambda k: results[k][0])
    feat_names = results[best_tag][1]
    idx = [LEGIT_FEATURES.index(f) for f in feat_names]
    print(f"\n  -> Selected: {best_tag} ({len(feat_names)} features)", flush=True)

    # ── Scale + final CV + holdout on the selected set ──
    X = X_all[:, idx]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    print("\n[2/3] Final 5-fold CV ...", flush=True)
    cv_acc, folds = cv_score(Xs, y, sw_map)
    print(f"    CV top-1: {cv_acc*100:.2f}%  folds: {[round(f*100,1) for f in folds]}", flush=True)

    print("\n[3/3] Holdout eval ...", flush=True)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.15, stratify=y, random_state=42)
    w_tr = np.array([sw_map[c] for c in y_tr])
    model = make_model()
    model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

    y_pred = model.predict(X_te)
    tr_acc = accuracy_score(y_tr, model.predict(X_tr))
    acc = accuracy_score(y_te, y_pred)
    top3 = (np.argsort(-model.predict_proba(X_te), axis=1)[:, :3] == y_te[:, None]).any(axis=1).mean()
    f1w = f1_score(y_te, y_pred, average='weighted')
    f1m = f1_score(y_te, y_pred, average='macro')

    print(f"\n  ===== FINAL (holdout, n={len(y_te)}) =====", flush=True)
    print(f"  Top-1 acc : {acc*100:.2f}%", flush=True)
    print(f"  Top-3 acc : {top3*100:.2f}%", flush=True)
    print(f"  Weighted F1: {f1w:.4f}", flush=True)
    print(f"  Macro F1   : {f1m:.4f}", flush=True)
    print(f"  5-fold CV  : {cv_acc*100:.2f}%", flush=True)
    print(f"  Train acc  : {tr_acc*100:.2f}%  gap: {(tr_acc-acc)*100:+.2f}pp", flush=True)

    print("\n  Feature importances:", flush=True)
    imp = sorted(zip(feat_names, model.feature_importances_), key=lambda x: -x[1])
    for name, score in imp:
        print(f"    {name:14s}: {score:.4f} {'#'*int(score*40)}", flush=True)

    print("\n  Worst 8 classes (holdout):", flush=True)
    scores = [accuracy_score(y_te[y_te == i], y_pred[y_te == i]) if (y_te == i).sum() > 0 else 0
              for i in range(N_CLASSES)]
    for cls in np.argsort(scores)[:8]:
        mask = y_te == cls
        ca = accuracy_score(y_te[mask], y_pred[mask]) if mask.sum() > 0 else 0
        name = le.inverse_transform([cls])[0][:22]
        print(f"    {name:24s}: {ca*100:5.1f}% (n={mask.sum()})", flush=True)

    # ── Save ──
    pickle.dump(model, open('models/xgb_crop_v11.pkl', 'wb'))
    pickle.dump(scaler, open('models/crop_scaler_v11.pkl', 'wb'))
    pickle.dump(feat_names, open('models/crop_featnames_v11.pkl', 'wb'))
    result = {
        'test_top1': round(float(acc), 6),
        'test_top3': round(float(top3), 6),
        'weighted_f1': round(float(f1w), 6),
        'macro_f1': round(float(f1m), 6),
        'cv_top1': round(float(cv_acc), 6),
        'cv_folds': [round(float(s), 6) for s in folds],
        'train_top1': round(float(tr_acc), 6),
        'overfit_gap': round(float(tr_acc - acc), 4),
        'n_features': int(len(feat_names)),
        'n_classes': N_CLASSES,
        'features': feat_names,
        'leak_free': True,
        'note': 'No label-derived categorical features. v10 98.2% was inflated '
                '~22pp by target leakage; this is the honest number.',
    }
    with open('outputs/crop_v11_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: models/xgb_crop_v11.pkl", flush=True)
    print(f"  Saved: outputs/crop_v11_metrics.json ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n** CROP v11 (LEAK-FREE): {acc*100:.2f}% top-1 | {top3*100:.2f}% top-3 | CV {cv_acc*100:.2f}% **", flush=True)
    print("=" * 62)


if __name__ == '__main__':
    train()