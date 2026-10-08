
import sys; sys.path.insert(0, 'src')
from preprocess import load_crop_data
import pandas as pd, numpy as np, pickle, os, json, time, warnings, gc
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
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
    eng['NPK_H'] = eng['NPK_total'] * eng['humidity']
    eng['HP'] = eng['humidity'] * eng['ph']
    eng['KP'] = eng['K'] * eng['ph']
    return eng.values, list(eng.columns)


def majority_pred(*arrays):
    return sci_mode(np.stack(arrays, axis=0), axis=0).mode[0]


def train():
    t0 = time.time()
    print("CROP MODEL - v8 | Unweighted + tuned", flush=True)

    X, y, feature_names = load_crop_data()
    X_eng, feat_names = engineer_features(X, feature_names)
    print(f"  {X_eng.shape}, classes: {len(np.unique(y))}, features: {X_eng.shape[1]}", flush=True)
    pickle.dump(feat_names, open('models/crop_feature_names.pkl', 'wb'))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_eng)
    pickle.dump(scaler, open('models/crop_scaler.pkl', 'wb'))

    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.15, stratify=y, random_state=42)

    models = {
        'XGBoost': XGBClassifier(
            n_estimators=600, max_depth=7, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            min_child_weight=3, gamma=0.2,
            reg_alpha=0.5, reg_lambda=3.0,
            random_state=42, tree_method='hist', verbosity=0, n_jobs=2),

        'LightGBM': lgbm.LGBMClassifier(
            n_estimators=600, max_depth=7, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            min_child_weight=3, reg_alpha=0.5, reg_lambda=3.0,
            random_state=42, n_jobs=2, verbose=-1),

        'CatBoost': CatBoostClassifier(
            iterations=600, depth=7, learning_rate=0.03,
            l2_leaf_reg=4.0, border_count=128,
            random_seed=42, verbose=0, thread_count=2),
    }

    trained = {}
    preds_te, preds_tr = [], []
    for name, m in models.items():
        print(f"  Training {name} ...", flush=True)
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])
        y_pt = m.predict(X_tr)
        y_pv = m.predict(X_te)
        print(f"    Train: {accuracy_score(y_tr, y_pt)*100:.2f}%  | Test: {accuracy_score(y_te, y_pv)*100:.2f}%", flush=True)
        trained[name] = m
        preds_tr.append(y_pt)
        preds_te.append(y_pv)
        gc.collect()

    y_ens_tr = majority_pred(*preds_tr)
    y_ens_te = majority_pred(*preds_te)
    acc_tr = accuracy_score(y_tr, y_ens_tr)
    acc_te = accuracy_score(y_te, y_ens_te)
    f1w = f1_score(y_te, y_ens_te, average='weighted')
    f1m = f1_score(y_te, y_ens_te, average='macro')

    print(f"\n  ===== FINAL =====", flush=True)
    print(f"  Train acc : {acc_tr*100:.2f}%", flush=True)
    print(f"  Test  acc : {acc_te*100:.2f}%", flush=True)
    print(f"  Gap       : {(acc_tr - acc_te)*100:+.2f}%", flush=True)
    print(f"  ENSEMBLE  : {acc_te*100:.2f}%", flush=True)

    pickle.dump(trained, open('models/crop_ensemble.pkl', 'wb'))
    pickle.dump(trained['XGBoost'], open('models/xgb_crop.pkl', 'wb'))
    print("  Saved models.", flush=True)

    n_classes = len(np.unique(y))
    print("\nWorst 5 classes:", flush=True)
    for cls in np.argsort([accuracy_score(y_te[y_te==i], y_ens_te[y_te==i]) if (y_te==i).sum()>0 else 0
                            for i in range(n_classes)])[:5]:
        mask = y_te == cls
        ca = accuracy_score(y_te[mask], y_ens_te[mask]) if mask.sum()>0 else 0
        print(f"  C{cls:2d}: {ca*100:.1f}% (n={mask.sum()})", flush=True)
        print(f"  Predicted as: {np.unique(y_ens_te[mask], return_counts=True)}", flush=True)

    print("\n", classification_report(y_te, y_ens_te, zero_division=0), flush=True)

    cm = confusion_matrix(y_te, y_ens_te)
    fig, ax = plt.subplots(figsize=(14, 10))
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(cm, display_labels=[f'C{i}' for i in range(n_classes)]).plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.title('Crop Ensemble')
    plt.tight_layout()
    plt.savefig('outputs/crop_confusion_matrix.png', dpi=150)
    print("Done!", flush=True)

if __name__ == '__main__':
    train()
