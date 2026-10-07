
import sys; sys.path.insert(0, 'src')
from preprocess import load_crop_data
import pandas as pd, numpy as np, pickle, os, warnings, json
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import optuna, optuna.logging as opl
opl.set_verbosity(opl.WARNING)
optuna.logging.disable_default_handler()

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt


def engineer_features(X, feature_names):
    n_samples = X.shape[0]
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
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_feats = poly.fit_transform(X)
    poly_names = poly.get_feature_names_out(feature_names)
    poly_df = pd.DataFrame(poly_feats, columns=poly_names)
    full = pd.concat([eng.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)
    feat_cols = list(full.columns)
    return full.values, feat_cols


def objective_xgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 600),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
        'use_label_encoder': False, 'eval_metric': 'mlogloss',
        'random_state': 42, 'tree_method': 'hist',
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds, trues = [], []
    for tr, va in cv.split(X, y):
        sm = SMOTE(random_state=42, k_neighbors=3)
        X_tr, y_tr = sm.fit_resample(X[tr], y[tr])
        m = XGBClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X[va], y[va])], verbose=False)
        preds.extend(m.predict(X[va]).tolist())
        trues.extend(y[va].tolist())
    return accuracy_score(trues, preds)


def objective_lgbm(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 600),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
        'random_state': 42, 'verbose': -1, 'n_jobs': 2,
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds, trues = [], []
    for tr, va in cv.split(X, y):
        sm = SMOTE(random_state=42, k_neighbors=3)
        X_tr, y_tr = sm.fit_resample(X[tr], y[tr])
        m = LGBMClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X[va], y[va])], callbacks=[], verbose=-1)
        preds.extend(m.predict(X[va]).tolist())
        trues.extend(y[va].tolist())
    return accuracy_score(trues, preds)


def objective_cb(trial, X, y):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 600),
        'depth': trial.suggest_int('depth', 4, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 7.0),
        'border_count': trial.suggest_int('border_count', 64, 255),
        'random_seed': 42, 'verbose': 0,
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds, trues = [], []
    for tr, va in cv.split(X, y):
        sm = SMOTE(random_state=42, k_neighbors=3)
        X_tr, y_tr = sm.fit_resample(X[tr], y[tr])
        m = CatBoostClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=(X[va], y[va]), verbose=False)
        preds.extend(m.predict(X[va]).tolist())
        trues.extend(y[va].tolist())
    return accuracy_score(trues, preds)


def train():
    print("=" * 60)
    print("CROP RECOMMENDATION - IMPROVED MODEL TRAINING")
    print("=" * 60)

    X, y, feature_names = load_crop_data()
    print(f"\nOriginal shape: X={X.shape}, y={y.shape}, classes={len(np.unique(y))}")

    # Feature engineering
    X_eng, feat_names = engineer_features(X, feature_names)
    print(f"Engineered features: {len(feat_names)} ({X_eng.shape[1]})")
    pickle.dump(feat_names, open('models/crop_feature_names.pkl', 'wb'))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_eng)
    pickle.dump(scaler, open('models/crop_scaler.pkl', 'wb'))

    # Step 1: Optuna hyperparameter tuning
    print("\n[1/5] Tuning XGBoost ...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(lambda t: objective_xgb(t, Xs, y), n_trials=40, show_progress_bar=False)
    best_xgb = study_xgb.best_params
    print(f"  XGBoost CV accuracy: {study_xgb.best_value*100:.2f}%")

    print("\n[2/5] Tuning LightGBM ...")
    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(lambda t: objective_lgbm(t, Xs, y), n_trials=40, show_progress_bar=False)
    best_lgbm = study_lgbm.best_params
    print(f"  LightGBM CV accuracy: {study_lgbm.best_value*100:.2f}%")

    print("\n[3/5] Tuning CatBoost ...")
    study_cb = optuna.create_study(direction='maximize')
    study_cb.optimize(lambda t: objective_cb(t, Xs, y), n_trials=40, show_progress_bar=False)
    best_cb = study_cb.best_params
    print(f"  CatBoost CV accuracy: {study_cb.best_value*100:.2f}%")

    # Step 2: Final ensemble on full train
    print("\n[4/5] Training final ensemble (SMOTE + 5-fold ...)")
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.15, stratify=y, random_state=42)
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
    print(f"  After SMOTE: X={X_sm.shape}, y={y_sm.shape}")

    xgb_m = XGBClassifier(
        **best_xgb, use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, tree_method='hist')
    lgbm_m = LGBMClassifier(**best_lgbm, random_state=42, verbose=-1, n_jobs=2)
    cb_m = CatBoostClassifier(**best_cb, random_seed=42, verbose=0)

    ens = VotingClassifier(
        estimators=[('xgb', xgb_m), ('lgbm', lgbm_m), ('cb', cb_m)],
        voting='soft', n_jobs=2)
    ens.fit(X_sm, y_sm)

    y_pred = ens.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1w = f1_score(y_te, y_pred, average='weighted')
    f1m = f1_score(y_te, y_pred, average='macro')
    print(f"\n  ENSEMBLE Test accuracy: {acc*100:.2f}%  |  weighted F1: {f1w:.4f}  |  macro F1: {f1m:.4f}")

    # Save
    pickle.dump(ens, open('models/crop_ensemble.pkl', 'wb'))
    print("\n  Saved: models/crop_ensemble.pkl, models/crop_scaler.pkl, models/crop_feature_names.pkl")

    # Step 3: Analysis
    print("\n[5/5] Detailed analysis ...")
    print(classification_report(y_te, y_pred))
    cm = confusion_matrix(y_te, y_pred)

    fig, ax = plt.subplots(figsize=(12, 10))
    class_names = np.unique(y)
    disp = ConfusionMatrixDisplay(cm, display_labels=[f'C{i}' for i in range(len(class_names))])
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title('Crop Ensemble: Confusion Matrix (Improved)')
    plt.tight_layout()
    plt.savefig('outputs/crop_confusion_matrix.png', dpi=150)
    print("Saved: outputs/crop_confusion_matrix.png")

    # Per-class accuracy
    per_class = {}
    for i in range(len(class_names)):
        mask = y_te == i
        if mask.sum() > 0:
            per_class[i] = accuracy_score(y_te[mask], y_pred[mask])
    worst = sorted(per_class.items(), key=lambda x: x[1])[:5]
    print("\nWorst 5 classes (need more data):")
    for cls, a in worst:
        print(f"  Class {cls}: {a*100:.1f}%")

    result = {
        'test_accuracy': round(float(acc), 6),
        'weighted_f1': round(float(f1w), 6),
        'macro_f1': round(float(f1m), 6),
        'xgb_cv_acc': round(float(study_xgb.best_value), 6),
        'lgbm_cv_acc': round(float(study_lgbm.best_value), 6),
        'cb_cv_acc': round(float(study_cb.best_value), 6),
        'n_features': int(X_eng.shape[1]),
        'n_classes': int(len(class_names)),
        'n_samples_after_smote': int(X_sm.shape[0]),
    }
    with open('outputs/crop_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: outputs/crop_metrics.json")
    print(f"\n{'='*60}")
    print(f"FINAL CROP MODEL: {acc*100:.2f}% accuracy")
    print(f"{'='*60}")

if __name__ == '__main__':
    train()
