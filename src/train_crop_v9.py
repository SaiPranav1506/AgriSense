import sys; sys.path.insert(0, 'src')
from preprocess import load_crop_data
import numpy as np, pickle, os, json, time, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import lightgbm as lgbm
from catboost import CatBoostClassifier

N_CLASSES = 46


def softened_weights(y):
    cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
    return dict(zip(np.unique(y), np.power(cw, 0.5)))  # mild, anti-overfit


def make_xgb():
    return XGBClassifier(
        n_estimators=900, max_depth=5, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6,
        min_child_weight=10, gamma=0.2,
        reg_alpha=0.5, reg_lambda=6.0, max_delta_step=5,
        random_state=42, tree_method='hist', verbosity=0, n_jobs=2)


def make_lgb():
    return lgbm.LGBMClassifier(
        n_estimators=900, max_depth=5, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
        reg_alpha=0.5, reg_lambda=6.0,
        random_state=42, n_jobs=2, verbose=-1)


def make_cb(wlist):
    return CatBoostClassifier(
        iterations=900, depth=5, learning_rate=0.03,
        l2_leaf_reg=8.0, border_count=128, class_weights=wlist,
        random_seed=42, verbose=0, thread_count=2)


def ens_proba(xgb, lgb, cb, Xv):
    p = np.zeros((len(Xv), N_CLASSES))
    p += xgb.predict_proba(Xv)
    p += lgb.predict_proba(Xv)
    p += cb.predict_proba(Xv)
    return p / 3.0


def train():
    t0 = time.time()
    print("=" * 60)
    print("CROP MODEL v9 | Regularized XGBoost + ensemble, strict 5-fold CV")
    print("=" * 60)

    X, y, feature_names = load_crop_data()
    print(f"  {X.shape}, classes: {len(np.unique(y))}, features: {X.shape[1]}", flush=True)
    assert X.shape[1] == 7, "expecting raw 7-feature scaler output (deployment-compatible)"

    sw_map = softened_weights(y)

    # ---- Honest generalization estimate: strict stratified 5-fold CV, XGBoost, weighted ----
    print("\n[1/2] Strict 5-fold CV (weighted, regularized XGBoost) ...", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds = np.zeros(len(y), dtype=int)
    fold_scores = []
    for fi, (tr, va) in enumerate(skf.split(X, y)):
        sw_tr = np.array([sw_map[c] for c in y[tr]])
        m = make_xgb()
        m.fit(X[tr], y[tr], sample_weight=sw_tr, verbose=False)
        va_pred = m.predict(X[va])
        cv_preds[va] = va_pred
        fold_scores.append(accuracy_score(y[va], va_pred))
        print(f"    fold{fi+1}: {fold_scores[-1]*100:.2f}%", flush=True)
    cv_acc = accuracy_score(y, cv_preds)
    print(f"    CV top-1: {cv_acc*100:.2f}%  (folds: {[round(s*100,2) for s in fold_scores]})", flush=True)

    # ---- Final models on train/val, evaluate on holdout test ----
    print("\n[2/2] Training ensemble (XGB + LightGBM + CatBoost) + soft vote ...", flush=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    sw_tr = np.array([sw_map[c] for c in y_tr])
    idxs = np.unique(y_tr)
    cb_weights = [sw_map[c] for c in idxs]

    xgb = make_xgb();  xgb.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
    lgb = make_lgb();  lgb.fit(X_tr, y_tr, sample_weight=sw_tr)
    cb = make_cb(cb_weights); cb.fit(X_tr, y_tr)

    p_te = ens_proba(xgb, lgb, cb, X_te)
    y_ens = np.argmax(p_te, axis=1)

    acc = accuracy_score(y_te, y_ens)
    top3 = (np.argsort(-p_te, axis=1)[:, :3] == y_te[:, None]).any(axis=1).mean()
    f1w = f1_score(y_te, y_ens, average='weighted')
    f1m = f1_score(y_te, y_ens, average='macro')

    print(f"\n  ===== FINAL (holdout, n={len(y_te)}) =====", flush=True)
    print(f"  Top-1 acc : {acc*100:.2f}%", flush=True)
    print(f"  Top-3 acc : {top3*100:.2f}%", flush=True)
    print(f"  Weighted  : {f1w:.4f}", flush=True)
    print(f"  Macro     : {f1m:.4f}", flush=True)
    print(f"  5-fold CV : {cv_acc*100:.2f}%", flush=True)
    print(f"  Overfit gap (holdout-train): -- (CV is the honest number)", flush=True)

    # Train accuracy for overfit diagnostic
    p_tr = ens_proba(xgb, lgb, cb, X_tr)
    tr_acc = accuracy_score(y_tr, np.argmax(p_tr, axis=1))
    print(f"  Train acc : {tr_acc*100:.2f}%  (gap vs holdout {(tr_acc-acc)*100:+.2f}%)", flush=True)

    # Worst classes
    print("\n  Worst 5 classes (holdout):", flush=True)
    for cls in np.argsort([accuracy_score(y_te[y_te == i], y_ens[y_te == i]) if (y_te == i).sum() > 0 else 0
                           for i in range(N_CLASSES)])[:5]:
        mask = y_te == cls
        ca = accuracy_score(y_te[mask], y_ens[mask]) if mask.sum() > 0 else 0
        print(f"    C{cls:2d}: {ca*100:.1f}% (n={mask.sum()})", flush=True)

    pickle.dump({'xgb': xgb, 'lgb': lgb, 'cb': cb,
                 'sw_map': sw_map, 'features': feature_names},
                open('models/crop_ensemble_v2.pkl', 'wb'))
    pickle.dump(xgb, open('models/xgb_crop_v2.pkl', 'wb'))
    print("\n  Saved: models/crop_ensemble_v2.pkl, models/xgb_crop_v2.pkl (production untouched)", flush=True)

    result = {
        'test_top1': round(float(acc), 6),
        'test_top3': round(float(top3), 6),
        'weighted_f1': round(float(f1w), 6),
        'macro_f1': round(float(f1m), 6),
        'cv_top1': round(float(cv_acc), 6),
        'cv_folds': [round(float(s), 6) for s in fold_scores],
        'train_top1': round(float(tr_acc), 6),
        'n_features': int(X.shape[1]),
        'n_classes': int(len(np.unique(y))),
    }
    with open('outputs/crop_v2_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: outputs/crop_v2_metrics.json ({time.time()-t0:.0f}s)", flush=True)
    print(f"** CROP v9: {acc*100:.2f}% top-1 | {top3*100:.2f}% top-3 | CV {cv_acc*100:.2f}% **", flush=True)
    print("=" * 60)


if __name__ == '__main__':
    train()