
import sys; sys.path.insert(0, 'src')
from preprocess import load_crop_data
import pandas as pd, numpy as np, pickle, os, json, time, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import mode as sci_mode

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
import lightgbm as lgbm
from catboost import CatBoostClassifier


def engineer_features(X, feature_names):
    eng = pd.DataFrame(X, columns=feature_names, dtype=float)
    eng['N_P_ratio'] = np.where(eng['P'] > 0, eng['N'] / eng['P'], 0.0)
    eng['N_K_ratio'] = np.where(eng['K'] > 0, eng['N'] / eng['K'], 0.0)
    eng['NPK_total'] = eng['N'] + eng['P'] + eng['K']
    eng['N_dev'] = (eng['N'] - 50).abs()
    eng['P_dev'] = (eng['P'] - 50).abs()
    eng['K_dev'] = (eng['K'] - 50).abs()
    eng['N_pct'] = np.where(eng['NPK_total'] > 0, eng['N'] / eng['NPK_total'], 0.0)
    eng['P_pct'] = np.where(eng['NPK_total'] > 0, eng['P'] / eng['NPK_total'], 0.0)
    eng['K_pct'] = np.where(eng['NPK_total'] > 0, eng['K'] / eng['NPK_total'], 0.0)
    eng['hum_temp'] = eng['humidity'] * eng['temperature']
    eng['rain_hum'] = eng['rainfall'] * eng['humidity']
    eng['ph_sq'] = eng['ph'] ** 2
    eng['temp_sq'] = eng['temperature'] ** 2
    eng['rain_ph'] = eng['rainfall'] * eng['ph']
    eng['NPK_H'] = eng['NPK_total'] * eng['humidity']
    eng['HP'] = eng['humidity'] * eng['ph']
    eng['KP'] = eng['K'] * eng['ph']
    return eng.values, list(eng.columns)


def majority_pred(preds):
    return sci_mode(np.array(preds), axis=0).mode[0]


def train():
    t0 = time.time()
    print("CROP MODEL - v7 | Balanced weights + Ensemble", flush=True)

    X, y, feature_names = load_crop_data()
    X_eng, feat_names = engineer_features(X, feature_names)
    print(f"  {X_eng.shape}, {len(np.unique(y))} classes, {X_eng.shape[1]} features", flush=True)
    pickle.dump(feat_names, open('models/crop_feature_names.pkl', 'wb'))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_eng)
    pickle.dump(scaler, open('models/crop_scaler.pkl', 'wb'))

    # Class weights (balanced) instead of SMOTE
    classes = np.unique(y)
    cw = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, cw))

    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.15, stratify=y, random_state=42)
    sample_weights = np.array([class_weight_dict[y] for y in y_tr])

    print("\n  Training XGBoost ...", flush=True)
    xgb = XGBClassifier(
        n_estimators=800, max_depth=7, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=3, gamma=0.2,
        reg_alpha=0.5, reg_lambda=3.0,
        random_state=42, tree_method='hist', verbosity=0, n_jobs=2)
    xgb.fit(X_tr, y_tr, sample_weight=sample_weights, eval_set=[(X_te, y_te)], verbose=False)
    y_xgb_tr = xgb.predict(X_tr)
    y_xgb_te = xgb.predict(X_te)
    print(f"    Train: {accuracy_score(y_tr, y_xgb_tr)*100:.2f}% | Test: {accuracy_score(y_te, y_xgb_te)*100:.2f}%", flush=True)

    print("  Training LightGBM ...", flush=True)
    lgb = lgbm.LGBMClassifier(
        n_estimators=800, max_depth=7, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=3, reg_alpha=0.5, reg_lambda=3.0,
        class_weight='balanced',
        random_state=42, n_jobs=2)
    lgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=-1, callbacks=[])
    y_lgb_tr = lgb.predict(X_tr)
    y_lgb_te = lgb.predict(X_te)
    print(f"    Train: {accuracy_score(y_tr, y_lgb_tr)*100:.2f}% | Test: {accuracy_score(y_te, y_lgb_te)*100:.2f}%", flush=True)

    print("  Training CatBoost ...", flush=True)
    cb = CatBoostClassifier(
        iterations=800, depth=7, learning_rate=0.03,
        l2_leaf_reg=4.0, border_count=128,
        class_weights=list(cw),
        random_seed=42, verbose=0, thread_count=2)
    cb.fit(X_tr, y_tr, eval_set=(X_te, y_te), verbose=False, plot=False)
    y_cb_tr = cb.predict(X_tr).astype(int).ravel()
    y_cb_te = cb.predict(X_te).astype(int).ravel()
    print(f"    Train: {accuracy_score(y_tr, y_cb_tr)*100:.2f}% | Test: {accuracy_score(y_te, y_cb_te)*100:.2f}%", flush=True)

    # Majority vote ensemble
    y_ens_tr = majority_pred([y_xgb_tr, y_lgb_tr, y_cb_tr])
    y_ens_te = majority_pred([y_xgb_te, y_lgb_te, y_cb_te])

    acc_tr = accuracy_score(y_tr, y_ens_tr)
    acc_te = accuracy_score(y_te, y_ens_te)
    f1w = f1_score(y_te, y_ens_te, average='weighted')
    f1m = f1_score(y_te, y_ens_te, average='macro')

    # Stratified 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    for tr_f, va_f in skf.split(Xs, y):
        xtr, xva = Xs[tr_f], Xs[va_f]
        ytr, yva = y[tr_f], y[va_f]
        sw_f = np.array([class_weight_dict[yy] for yy in ytr])
        xm = XGBClassifier(n_estimators=800, max_depth=7, learning_rate=0.03, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=3, gamma=0.2,
            reg_alpha=0.5, reg_lambda=3.0, random_state=42, tree_method='hist', verbosity=0, n_jobs=2)
        xm.fit(xtr, ytr, sample_weight=sw_f, verbose=False)
        lm = lgbm.LGBMClassifier(n_estimators=800, max_depth=7, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            reg_alpha=0.5, reg_lambda=3.0, class_weight='balanced',
            random_state=42, n_jobs=2)
        lm.fit(xtr, ytr, verbose=-1, callbacks=[])
        cm = CatBoostClassifier(iterations=800, depth=7, learning_rate=0.03, l2_leaf_reg=4.0,
            class_weights=list(cw), random_seed=42, verbose=0, thread_count=2)
        cm.fit(xtr, ytr, verbose=False, plot=False)
        sp = np.stack([xm.predict(Xs[va_f]), lm.predict(Xs[va_f]),
                        cm.predict(Xs[va_f]).astype(int).ravel()], axis=0)
        cv_accs.append(accuracy_score(y[va_f], sci_mode(sp, axis=0).mode[0]))

    cv_mean = np.mean(cv_accs)
    print(f"\n  ===== FINAL RESULTS =====", flush=True)
    print(f"  Train acc  : {acc_tr*100:.2f}%", flush=True)
    print(f"  Test  acc  : {acc_te*100:.2f}%", flush=True)
    print(f"  5-fold CV  : {cv_mean*100:.2f}%  (+/- {np.std(cv_accs)*100:.1f}%)", flush=True)
    print(f"  Overfit gap: {(acc_tr - acc_te)*100:+.2f}%", flush=True)
    print(f"  XGBoost    : {accuracy_score(y_te, y_xgb_te)*100:.2f}%", flush=True)
    print(f"  LightGBM   : {accuracy_score(y_te, y_lgb_te)*100:.2f}%", flush=True)
    print(f"  CatBoost   : {accuracy_score(y_te, y_cb_te)*100:.2f}%", flush=True)
    print(f"  ENSEMBLE   : {acc_te*100:.2f}%", flush=True)
    print(f"  Weighted F1: {f1w:.4f}  |  Macro F1: {f1m:.4f}", flush=True)

    pickle.dump({'xgb': xgb, 'lgb': lgb, 'cb': cb, 'feat_names': feat_names, 'scaler': scaler},
                open('models/crop_ensemble.pkl', 'wb'))
    pickle.dump(xgb, open('models/xgb_crop.pkl', 'wb'))
    print("\n  Saved: models/crop_ensemble.pkl, models/xgb_crop.pkl", flush=True)

    n_classes = len(classes)
    print("\nWorst 5 classes:", flush=True)
    worst = np.argsort([accuracy_score(y_te[y_te==i], y_ens_te[y_te==i]) if (y_te==i).sum() > 0 else 0
                        for i in range(n_classes)])[:5]
    for cls in worst:
        mask = y_te == cls
        ca = accuracy_score(y_te[mask], y_ens_te[mask]) if mask.sum() > 0 else 0
        print(f"  Class {cls:2d}: {ca*100:.1f}% (n={mask.sum()})", flush=True)

    print("\n", classification_report(y_te, y_ens_te, zero_division=0), flush=True)

    cm = confusion_matrix(y_te, y_ens_te)
    fig, ax = plt.subplots(figsize=(14, 10))
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(cm, display_labels=[f'C{i}' for i in range(n_classes)]).plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.title('Crop Ensemble - Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/crop_confusion_matrix.png', dpi=150)
    print("Saved: outputs/crop_confusion_matrix.png", flush=True)

    result = {
        'test_accuracy': round(float(acc_te), 6),
        'train_accuracy': round(float(acc_tr), 6),
        'cv_accuracy': round(float(cv_mean), 6),
        'cv_std': round(float(np.std(cv_accs)), 6),
        'overfit_gap': round(float(acc_tr - acc_te), 4),
        'weighted_f1': round(float(f1w), 6),
        'macro_f1': round(float(f1m), 6),
        'xgb_acc': round(float(accuracy_score(y_te, y_xgb_te)), 6),
        'lgbm_acc': round(float(accuracy_score(y_te, y_lgb_te)), 6),
        'cb_acc': round(float(accuracy_score(y_te, y_cb_te)), 6),
        'n_features': int(X_eng.shape[1]),
        'n_classes': int(n_classes),
    }
    with open('outputs/crop_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"** CROP MODEL: {acc_te*100:.2f}% test | CV: {cv_mean*100:.2f}% **", flush=True)
    print("=" * 60)

if __name__ == '__main__':
    train()
