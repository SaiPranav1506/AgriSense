
import sys; sys.path.insert(0, 'src')
from preprocess import load_crop_data
import pandas as pd, numpy as np, pickle, os, json, time, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
import lightgbm as lgbm
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE


def engineer_features(X, feature_names):
    eng = pd.DataFrame(X, columns=feature_names)
    eng['N_P_ratio'] = np.where(eng['P'] > 0, eng['N'] / eng['P'], 0.0)
    eng['N_K_ratio'] = np.where(eng['K'] > 0, eng['N'] / eng['K'], 0.0)
    eng['NPK_total'] = eng['N'] + eng['P'] + eng['K']
    eng['N_pct'] = np.where(eng['NPK_total'] > 0, eng['N'] / eng['NPK_total'], 0.0)
    eng['P_pct'] = np.where(eng['NPK_total'] > 0, eng['P'] / eng['NPK_total'], 0.0)
    eng['K_pct'] = np.where(eng['NPK_total'] > 0, eng['K'] / eng['NPK_total'], 0.0)
    eng['NPK_sum'] = eng['NPK_total']
    eng['NPK_N'] = (eng['N'] - 40).clip(lower=0)
    eng['NPK_P'] = (eng['P'] - 40).clip(lower=0)
    eng['NPK_K'] = (eng['K'] - 40).clip(lower=0)
    eng['hum_temp'] = eng['humidity'] * eng['temperature']
    eng['rain_sq'] = eng['rainfall'] ** 2
    eng['ph_dev'] = (eng['ph'] - 6.5).abs()
    eng['ph_ph'] = eng['ph'] * eng['ph']
    eng['rain_hum'] = eng['rainfall'] * eng['humidity']
    eng['NPK_rain'] = eng['NPK_total'] * eng['rainfall'] / 200.0
    eng['HP'] = eng['humidity'] * eng['ph']
    eng['KP'] = eng['K'] * eng['ph']
    return eng.values, list(eng.columns)


def majority_pred(preds):
    from scipy.stats import mode as sci_mode
    return sci_mode(np.array(preds), axis=0).mode[0]


def train():
    t0 = time.time()
    print("CROP MODEL - FAST v5 | Feature Engineering + SMOTE + Ensemble", flush=True)

    X, y, feature_names = load_crop_data()
    X_eng, feat_names = engineer_features(X, feature_names)
    print(f"  {X_eng.shape} samples x features, {len(np.unique(y))} classes", flush=True)
    pickle.dump(feat_names, open('models/crop_feature_names.pkl', 'wb'))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_eng)
    pickle.dump(scaler, open('models/crop_scaler.pkl', 'wb'))

    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.15, stratify=y, random_state=42)

    print("  Applying SMOTE (once, on train only) ...", flush=True)
    sm = SMOTE(random_state=42, k_neighbors=4)
    X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
    print(f"  After SMOTE: {X_sm.shape}", flush=True)

    # Train each model with carefully chosen hyperparameters
    print("\n  Training XGBoost ...", flush=True)
    xgb = XGBClassifier(
        n_estimators=1200, max_depth=12, learning_rate=0.02,
        subsample=0.9, colsample_bytree=0.9,
        min_child_weight=1, gamma=0.05,
        reg_alpha=0.3, reg_lambda=2.0,
        random_state=42, tree_method='hist', verbosity=0)
    xgb.fit(X_sm, y_sm, eval_set=[(X_te, y_te)], verbose=False)
    y_xgb_tr = xgb.predict(X_tr)
    y_xgb_te = xgb.predict(X_te)
    print(f"    Train: {accuracy_score(y_tr, y_xgb_tr)*100:.2f}% | Test: {accuracy_score(y_te, y_xgb_te)*100:.2f}%", flush=True)

    print("  Training LightGBM ...", flush=True)
    lgb = lgbm.LGBMClassifier(
        n_estimators=1200, max_depth=12, learning_rate=0.02,
        subsample=0.9, colsample_bytree=0.9,
        min_child_weight=1, reg_alpha=0.3, reg_lambda=2.0,
        random_state=42, verbose=-1, n_jobs=2)
    lgb.fit(X_sm, y_sm, eval_set=[(X_te, y_te)], verbose=-1)
    y_lgb_tr = lgb.predict(X_tr)
    y_lgb_te = lgb.predict(X_te)
    print(f"    Train: {accuracy_score(y_tr, y_lgb_tr)*100:.2f}% | Test: {accuracy_score(y_te, y_lgb_te)*100:.2f}%", flush=True)

    print("  Training CatBoost ...", flush=True)
    cb = CatBoostClassifier(
        iterations=1200, depth=10, learning_rate=0.02,
        l2_leaf_reg=3.5, border_count=255,
        random_seed=42, verbose=0, thread_count=2)
    cb.fit(X_sm, y_sm, eval_set=(X_te, y_te), verbose=False)
    y_cb_tr = cb.predict(X_tr).astype(int).ravel()
    y_cb_te = cb.predict(X_te).astype(int).ravel()
    cb_tr_acc = accuracy_score(y_tr, y_cb_tr)
    cb_te_acc = accuracy_score(y_te, y_cb_te)
    print(f"    Train: {cb_tr_acc*100:.2f}% | Test: {cb_te_acc*100:.2f}%", flush=True)

    # Ensemble
    y_ens_tr = majority_pred([y_xgb_tr, y_lgb_tr, y_cb_tr])
    y_ens_te = majority_pred([y_xgb_te, y_lgb_te, y_cb_te])

    acc_tr = accuracy_score(y_tr, y_ens_tr)
    acc_te = accuracy_score(y_te, y_ens_te)
    f1w = f1_score(y_te, y_ens_te, average='weighted')
    f1m = f1_score(y_te, y_ens_te, average='macro')

    print(f"\n  ===== FINAL RESULTS =====", flush=True)
    print(f"  Train acc  : {acc_tr*100:.2f}%", flush=True)
    print(f"  Test  acc  : {acc_te*100:.2f}%", flush=True)
    print(f"  Overfit gap: {(acc_tr - acc_te)*100:+.2f}%", flush=True)
    print(f"  XGBoost    : {accuracy_score(y_te, y_xgb_te)*100:.2f}%", flush=True)
    print(f"  LightGBM   : {accuracy_score(y_te, y_lgb_te)*100:.2f}%", flush=True)
    print(f"  CatBoost   : {cb_te_acc*100:.2f}%", flush=True)
    print(f"  Ensemble   : {acc_te*100:.2f}%", flush=True)
    print(f"  Weighted F1: {f1w:.4f}  |  Macro F1: {f1m:.4f}", flush=True)

    pickle.dump({'xgb': xgb, 'lgb': lgb, 'cb': cb}, open('models/crop_ensemble.pkl', 'wb'))
    pickle.dump(xgb, open('models/xgb_crop.pkl', 'wb'))
    print("\n  Saved: models/crop_ensemble.pkl, models/xgb_crop.pkl", flush=True)

    print("\nWorst 5 classes:", flush=True)
    n_classes = len(np.unique(y))
    for cls in np.argsort([accuracy_score(y_te[y_te==i], y_ens_te[y_te==i]) if (y_te==i).sum() > 0 else 0
                            for i in range(n_classes)])[:5]:
        mask = y_te == cls
        ca = accuracy_score(y_te[mask], y_ens_te[mask]) if mask.sum() > 0 else 0
        print(f"  Class {cls:2d}: {ca*100:.1f}% (n={mask.sum()})", flush=True)

    cm = confusion_matrix(y_te, y_ens_te)
    fig, ax = plt.subplots(figsize=(14, 10))
    ConfusionMatrixDisplay(cm, display_labels=[f'C{i}' for i in range(n_classes)]).plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.title('Crop Ensemble - Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/crop_confusion_matrix.png', dpi=150)

    result = {
        'test_accuracy': round(float(acc_te), 6),
        'train_accuracy': round(float(acc_tr), 6),
        'overfit_gap': round(float(acc_tr - acc_te), 4),
        'weighted_f1': round(float(f1w), 6),
        'macro_f1': round(float(f1m), 6),
        'xgb_acc': round(float(accuracy_score(y_te, y_xgb_te)), 6),
        'lgbm_acc': round(float(accuracy_score(y_te, y_lgb_te)), 6),
        'cb_acc': round(float(cb_te_acc), 6),
        'n_features': int(X_eng.shape[1]),
        'n_classes': int(n_classes),
    }
    with open('outputs/crop_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"** CROP MODEL: {acc_te*100:.2f}% test accuracy **")
    print("=" * 60)

if __name__ == '__main__':
    train()
