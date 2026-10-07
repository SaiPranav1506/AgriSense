import sys; sys.path.insert(0, 'src')
from crop_features import add_features
import numpy as np, pickle, os, json, time, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

N_CLASSES = 46


def softened_weights(y):
    cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
    return dict(zip(np.unique(y), np.power(cw, 0.5)))


def train():
    t0 = time.time()
    print("=" * 60)
    print("CROP MODEL v10 | Enhanced features (21 features)")
    print("=" * 60, flush=True)

    import pandas as pd
    df = pd.read_csv('data/raw/Crop_recommendation_Extended.csv')
    X, feat_names = add_features(df)

    le = pickle.load(open('models/label_encoder.pkl', 'rb'))
    y = le.transform(df['label'].values).astype(np.int32)
    print(f"  X={X.shape}, y={y.shape}, classes={N_CLASSES}, features={len(feat_names)}", flush=True)
    print(f"  Features: {feat_names}", flush=True)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pickle.dump(scaler, open('models/crop_scaler_v10.pkl', 'wb'))
    pickle.dump(feat_names, open('models/crop_featnames_v10.pkl', 'wb'))

    sw_map = softened_weights(y)

    # ── 5-fold CV ──
    print("\n[1/3] 5-fold CV ...", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds = np.zeros(len(y), dtype=int)
    fold_scores = []
    for fi, (tr, va) in enumerate(skf.split(Xs, y)):
        sw_tr = np.array([sw_map[c] for c in y[tr]])
        m = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.75, colsample_bytree=0.7,
            min_child_weight=8, gamma=0.2,
            reg_alpha=0.5, reg_lambda=6.0, max_delta_step=5,
            random_state=42, tree_method='hist', verbosity=0, n_jobs=2)
        m.fit(Xs[tr], y[tr], sample_weight=sw_tr, verbose=False)
        cv_preds[va] = m.predict(Xs[va])
        fold_scores.append(accuracy_score(y[va], cv_preds[va]))
        print(f"    fold{fi+1}: {fold_scores[-1]*100:.2f}%", flush=True)
        del m
    cv_acc = accuracy_score(y, cv_preds)
    print(f"    CV top-1: {cv_acc*100:.2f}%  folds: {[round(s*100,1) for s in fold_scores]}", flush=True)

    # ── Train/val split ──
    print("\n[2/3] Training final model ...", flush=True)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.15, stratify=y, random_state=42)
    sw_tr = np.array([sw_map[c] for c in y_tr])

    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.75, colsample_bytree=0.7,
        min_child_weight=8, gamma=0.2,
        reg_alpha=0.5, reg_lambda=6.0, max_delta_step=5,
        random_state=42, tree_method='hist', verbosity=0, n_jobs=2)
    model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
    print("  Training complete.", flush=True)

    # ── Evaluate ──
    print("\n[3/3] Evaluating ...", flush=True)
    y_pred = model.predict(X_te)
    y_pred_tr = model.predict(X_tr)
    acc = accuracy_score(y_te, y_pred)
    tr_acc = accuracy_score(y_tr, y_pred_tr)
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

    # Feature importances
    print("\n  Feature importances (XGBoost):", flush=True)
    imp = sorted(zip(feat_names, model.feature_importances_), key=lambda x: -x[1])
    for name, score in imp:
        bar = '█' * int(score * 40)
        print(f"    {name:20s}: {score:.4f} {bar}", flush=True)

    # Worst 8 classes
    print("\n  Worst 8 classes (holdout):", flush=True)
    scores = [accuracy_score(y_te[y_te==i], y_pred[y_te==i]) if (y_te==i).sum()>0 else 0
              for i in range(N_CLASSES)]
    for cls in np.argsort(scores)[:8]:
        mask = y_te == cls
        ca = accuracy_score(y_te[mask], y_pred[mask]) if mask.sum()>0 else 0
        name = le.inverse_transform([cls])[0][:22]
        print(f"    {name:24s}: {ca*100:5.1f}% (n={mask.sum()})", flush=True)

    # Save
    pickle.dump(model, open('models/xgb_crop_v10.pkl', 'wb'))
    result = {
        'test_top1': round(float(acc), 6),
        'test_top3': round(float(top3), 6),
        'weighted_f1': round(float(f1w), 6),
        'macro_f1': round(float(f1m), 6),
        'cv_top1': round(float(cv_acc), 6),
        'cv_folds': [round(float(s), 6) for s in fold_scores],
        'train_top1': round(float(tr_acc), 6),
        'overfit_gap': round(float(tr_acc - acc), 4),
        'n_features': int(len(feat_names)),
        'n_classes': N_CLASSES,
        'features': feat_names,
        'n_estimators': 500,
        'max_depth': 6,
    }
    with open('outputs/crop_v10_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: models/xgb_crop_v10.pkl", flush=True)
    print(f"  Saved: outputs/crop_v10_metrics.json ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n** CROP v10: {acc*100:.2f}% top-1 | {top3*100:.2f}% top-3 | CV {cv_acc*100:.2f}% **", flush=True)
    print("=" * 60)


if __name__ == '__main__':
    train()