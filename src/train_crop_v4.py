
import sys; sys.path.insert(0, 'src')
from preprocess import load_crop_data
import pandas as pd, numpy as np, pickle, os, json, time, warnings
warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
import lightgbm as lgbm
from catboost import CatBoostClassifier


def engineer_features(X, feature_names):
    eng = pd.DataFrame(X, columns=feature_names)
    eng['N_P_ratio'] = np.where(eng['P'] > 0, eng['N'] / eng['P'], 0)
    eng['N_K_ratio'] = np.where(eng['K'] > 0, eng['N'] / eng['K'], 0)
    eng['P_K_ratio'] = np.where(eng['K'] > 0, eng['P'] / eng['K'], 0)
    eng['NPK_total'] = eng['N'] + eng['P'] + eng['K']
    eng['N_pct'] = np.where(eng['NPK_total'] > 0, eng['N'] / eng['NPK_total'], 0)
    eng['P_pct'] = np.where(eng['NPK_total'] > 0, eng['P'] / eng['NPK_total'], 0)
    eng['K_pct'] = np.where(eng['NPK_total'] > 0, eng['K'] / eng['NPK_total'], 0)
    eng['hum_temp'] = eng['humidity'] * eng['temperature']
    eng['rain_hum'] = eng['rainfall'] * eng['humidity']
    eng['ph_sq'] = eng['ph'] ** 2
    eng['temp_sq'] = eng['temperature'] ** 2
    eng['rain_ph'] = eng['rainfall'] * eng['ph']
    eng['NPK_H'] = eng['NPK_total'] * eng['humidity']
    return eng.values, list(eng.columns)


def train():
    t0 = time.time()
    print("CROP MODEL - FAST IMPROVED v4", flush=True)

    X, y, feature_names = load_crop_data()
    X_eng, feat_names = engineer_features(X, feature_names)
    print(f"Shape: {X_eng.shape}, classes: {len(np.unique(y))}, features: {X_eng.shape[1]}", flush=True)
    pickle.dump(feat_names, open('models/crop_feature_names.pkl', 'wb'))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_eng)
    pickle.dump(scaler, open('models/crop_scaler.pkl', 'wb'))

    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.15, stratify=y, random_state=42)

    print("Training XGBoost ...", flush=True)
    xgb = XGBClassifier(
        n_estimators=800, max_depth=10, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=1, gamma=0,
        reg_alpha=0.3, reg_lambda=2.0,
        random_state=42, tree_method='hist', verbosity=0,
        n_jobs=2)
    xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    y_xgb = xgb.predict(X_te)
    print(f"  XGBoost test: {accuracy_score(y_te, y_xgb)*100:.2f}%", flush=True)

    print("Training LightGBM ...", flush=True)
    lgb = lgbm.LGBMClassifier(
        n_estimators=800, max_depth=10, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=1, reg_alpha=0.3, reg_lambda=2.0,
        random_state=42, verbose=-1, n_jobs=2)
    lgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=-1)
    y_lgb = lgb.predict(X_te)
    print(f"  LightGBM test: {accuracy_score(y_te, y_lgb)*100:.2f}%", flush=True)

    print("Training CatBoost ...", flush=True)
    cb = CatBoostClassifier(
        iterations=800, depth=9, learning_rate=0.03,
        l2_leaf_reg=3.0, border_count=128,
        random_seed=42, verbose=0, thread_count=2)
    cb.fit(X_tr, y_tr, eval_set=(X_te, y_te), verbose=False)
    y_cb = cb.predict(X_te).astype(int).ravel()
    print(f"  CatBoost test: {accuracy_score(y_te, y_cb)*100:.2f}%", flush=True)

    # Ensemble by majority vote
    from scipy.stats import mode as scimode
    stacked = np.stack([y_xgb, y_lgb, y_cb], axis=0)
    y_ens = scimode(stacked, axis=0).mode[0]

    acc = accuracy_score(y_te, y_ens)
    f1w = f1_score(y_te, y_ens, average='weighted')
    f1m = f1_score(y_te, y_ens, average='macro')

    print(f"\nEnsemble test: {acc*100:.2f}%  weighted_F1={f1w:.4f}  macro_F1={f1m:.4f}", flush=True)

    # Cross-validation via simple 3-fold majority vote
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_preds = np.zeros(len(y), dtype=int)
    for tr, va in skf.split(Xs, y):
        x = XGBClassifier(n_estimators=800, max_depth=10, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=1, gamma=0,
            reg_alpha=0.3, reg_lambda=2.0, random_state=42, tree_method='hist', verbosity=0, n_jobs=2).fit(Xs[tr], y[tr])
        l = lgbm.LGBMClassifier(n_estimators=800, max_depth=10, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=1,
            reg_alpha=0.3, reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=2).fit(Xs[tr], y[tr])
        c = CatBoostClassifier(iterations=800, depth=9, learning_rate=0.03, l2_leaf_reg=3.0,
            random_seed=42, verbose=0, thread_count=2).fit(Xs[tr], y[tr])
        s = np.stack([x.predict(Xs[va]), l.predict(Xs[va]), c.predict(Xs[va]).astype(int).ravel()], axis=0)
        cv_preds[va] = scimode(s, axis=0).mode[0]

    cv_acc = accuracy_score(y, cv_preds)
    print(f"  3-fold CV    : {cv_acc*100:.2f}%  (overfit check: gap={(acc-cv_acc)*100:+.2f}%)", flush=True)

    # Save best single model (XGBoost) as the primary model for backward compat
    # Also save the ensemble
    ens_models = {'xgb': xgb, 'lgb': lgb, 'cb': cb}
    pickle.dump(ens_models, open('models/crop_ensemble.pkl', 'wb'))
    pickle.dump(xgb, open('models/xgb_crop.pkl', 'wb'))
    print("Saved: models/crop_ensemble.pkl, models/xgb_crop.pkl", flush=True)

    print("\nPer-class accuracy (worst 5):", flush=True)
    for cls in np.argsort([accuracy_score(y_te[y_te==i], y_ens[y_te==i]) if (y_te==i).sum() > 0 else 0
                            for i in range(len(np.unique(y)))])[:5]:
        mask = y_te == cls
        ca = accuracy_score(y_te[mask], y_ens[mask]) if mask.sum() > 0 else 0
        print(f"  Class {cls:2d}: {ca*100:.1f}% (n={mask.sum()})", flush=True)

    result = {
        'test_accuracy': round(float(acc), 6),
        'weighted_f1': round(float(f1w), 6),
        'macro_f1': round(float(f1m), 6),
        'cv_accuracy': round(float(cv_acc), 6),
        'overfit_gap': round(float(acc - cv_acc), 4),
        'xgb_acc': round(float(accuracy_score(y_te, y_xgb)), 6),
        'lgbm_acc': round(float(accuracy_score(y_te, y_lgb)), 6),
        'cb_acc': round(float(accuracy_score(y_te, y_cb)), 6),
        'n_features': int(X_eng.shape[1]),
        'n_classes': int(len(np.unique(y))),
    }
    with open('outputs/crop_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"** CROP MODEL: {acc*100:.2f}% test | {cv_acc*100:.2f}% CV **")
    print("=" * 60)

if __name__ == '__main__':
    train()
