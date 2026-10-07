
import sys; sys.path.insert(0, 'src')
from preprocess import load_crop_data
import pandas as pd, numpy as np, pickle, os, warnings, json, time
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import optuna, optuna.logging as opl
optuna.logging.disable_default_handler()

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build_features(X, feature_names):
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
    feat_cols = list(eng.columns)
    return eng.values, feat_cols


def cv_score_xgb(params, X, y):
    params_use = {k: v for k, v in params.items() if k != 'verbosity'}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds, trues = [], []
    for tr, va in cv.split(X, y):
        sm = SMOTE(random_state=42, k_neighbors=3)
        X_tr, y_tr = sm.fit_resample(X[tr], y[tr])
        m = XGBClassifier(**params_use, verbosity=0, random_state=42, tree_method='hist')
        m.fit(X_tr, y_tr, eval_set=[(X[va], y[va])], verbose=False)
        preds.extend(m.predict(X[va]).tolist())
        trues.extend(y[va].tolist())
    return accuracy_score(trues, preds)

def cv_score_lgbm(params, X, y):
    params_use = {k: v for k, v in params.items() if k != 'verbosity'}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds, trues = [], []
    for tr, va in cv.split(X, y):
        sm = SMOTE(random_state=42, k_neighbors=3)
        X_tr, y_tr = sm.fit_resample(X[tr], y[tr])
        m = LGBMClassifier(**params_use, random_state=42, verbose=-1, n_jobs=2)
        m.fit(X_tr, y_tr, eval_set=[(X[va], y[va])], callbacks=[], verbose=-1)
        preds.extend(m.predict(X[va]).tolist())
        trues.extend(y[va].tolist())
    return accuracy_score(trues, preds)

def cv_score_cb(params, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds, trues = [], []
    for tr, va in cv.split(X, y):
        sm = SMOTE(random_state=42, k_neighbors=3)
        X_tr, y_tr = sm.fit_resample(X[tr], y[tr])
        m = CatBoostClassifier(**params, random_seed=42, verbose=0)
        m.fit(X_tr, y_tr, eval_set=(X[va], y[va]), verbose=False)
        preds.extend(m.predict(X[va]).tolist())
        trues.extend(y[va].tolist())
    return accuracy_score(trues, preds)


def tune_model(make_objective, name, n_trials=25):
    study = optuna.create_study(direction='maximize')
    study.optimize(make_objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def train():
    t0 = time.time()
    print("=" * 60)
    print("CROP RECOMMENDATION - FAST IMPROVED TRAINING")
    print("=" * 60)

    X, y, feature_names = load_crop_data()
    X_eng, feat_names = build_features(X, feature_names)
    print(f"Features: {X_eng.shape[1]} (engineered from {X.shape[1]})")
    pickle.dump(feat_names, open('models/crop_feature_names.pkl', 'wb'))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_eng)
    pickle.dump(scaler, open('models/crop_scaler.pkl', 'wb'))

    # Tune each model (20 trials each)
    print("\n[1/3] Tuning XGBoost (20 trials)...", flush=True)
    bx, vx = tune_model(lambda t: cv_score_xgb({
        'n_estimators': t.suggest_int('n_estimators', 300, 600),
        'max_depth': t.suggest_int('max_depth', 4, 10),
        'learning_rate': t.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': t.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': t.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': t.suggest_int('min_child_weight', 1, 10),
        'gamma': t.suggest_float('gamma', 0, 1.0),
        'reg_alpha': t.suggest_float('reg_alpha', 0, 2.0),
        'reg_lambda': t.suggest_float('reg_lambda', 1.0, 5.0),
    }, Xs, y), 'xgb', 20)
    del bx['verbosity']
    print(f"  XGBoost 5-fold CV: {vx*100:.2f}%", flush=True)

    print("\n[2/3] Tuning LightGBM (20 trials)...", flush=True)
    bl, vl = tune_model(lambda t: cv_score_lgbm({
        'n_estimators': t.suggest_int('n_estimators', 300, 600),
        'max_depth': t.suggest_int('max_depth', 4, 10),
        'learning_rate': t.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': t.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': t.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': t.suggest_int('min_child_weight', 1, 15),
        'reg_alpha': t.suggest_float('reg_alpha', 0, 2.0),
        'reg_lambda': t.suggest_float('reg_lambda', 1.0, 5.0),
    }, Xs, y), 'lgbm', 20)
    del bl['verbosity']
    print(f"  LightGBM 5-fold CV: {vl*100:.2f}%", flush=True)

    print("\n[3/3] Tuning CatBoost (20 trials)...", flush=True)
    bc, vc = tune_model(lambda t: cv_score_cb({
        'iterations': t.suggest_int('iterations', 300, 600),
        'depth': t.suggest_int('depth', 4, 9),
        'learning_rate': t.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'l2_leaf_reg': t.suggest_float('l2_leaf_reg', 1.0, 7.0),
        'border_count': t.suggest_int('border_count', 64, 255),
    }, Xs, y), 'catboost', 20)
    print(f"  CatBoost 5-fold CV: {vc*100:.2f}%", flush=True)

    # Train final ensemble with SMOTE on held-out 15% test
    print("\nTraining final ensemble on held-out test set ...", flush=True)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.15, stratify=y, random_state=42)
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_sm, y_sm = sm.fit_resample(X_tr, y_tr)

    xgb_m = XGBClassifier(**bx, verbosity=0, random_state=42, tree_method='hist')
    lgbm_m = LGBMClassifier(**bl, random_state=42, verbose=-1, n_jobs=2)
    cb_m = CatBoostClassifier(**bc, random_seed=42, verbose=0)

    ens = VotingClassifier(
        estimators=[('xgb', xgb_m), ('lgbm', lgbm_m), ('cb', cb_m)],
        voting='soft', n_jobs=2, weights=[1,1,1])
    ens.fit(X_sm, y_sm)

    y_pred = ens.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1w = f1_score(y_te, y_pred, average='weighted')
    f1m = f1_score(y_te, y_pred, average='macro')

    # Stratified 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds = cross_val_predict(ens, Xs, y, cv=cv)
    cv_acc = accuracy_score(y, cv_preds)

    print(f"\n  Ensemble test accuracy : {acc*100:.2f}%")
    print(f"  Weighted F1            : {f1w:.4f}")
    print(f"  Macro F1               : {f1m:.4f}")
    print(f"  5-fold CV accuracy     : {cv_acc*100:.2f}%")

    pickle.dump(ens, open('models/crop_ensemble.pkl', 'wb'))
    print("\n  Saved: models/crop_ensemble.pkl")

    print("\n" + classification_report(y_te, y_pred))
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(cm, display_labels=[f'C{i}' for i in range(len(np.unique(y)))]).plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title('Crop Ensemble: Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/crop_confusion_matrix.png', dpi=150)

    per_class = {i: accuracy_score(y_te[y_te==i], y_pred[y_te==i]) for i in range(len(np.unique(y)))}
    worst = sorted(per_class.items(), key=lambda x: x[1])[:5]
    print("Worst 5 classes:")
    for cls, a in worst:
        print(f"  Class {cls}: {a*100:.1f}%")

    result = {
        'test_accuracy': round(float(acc), 6),
        'weighted_f1': round(float(f1w), 6),
        'macro_f1': round(float(f1m), 6),
        'cv_accuracy': round(float(cv_acc), 6),
        'xgb_cv': round(float(vx), 6),
        'lgbm_cv': round(float(vl), 6),
        'cb_cv': round(float(vc), 6),
        'n_features': int(X_eng.shape[1]),
        'n_classes': int(len(np.unique(y))),
    }
    json.dump(result, open('outputs/crop_metrics.json', 'w'), indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"FINAL CROP MODEL: {acc*100:.2f}% accuracy | CV: {cv_acc*100:.2f}%")
    print("=" * 60)

if __name__ == '__main__':
    train()
