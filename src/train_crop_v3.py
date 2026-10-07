
import sys; sys.path.insert(0, 'src')
from preprocess import load_crop_data
import pandas as pd, numpy as np, pickle, os, json, time
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.ensemble import VotingClassifier, StackingClassifier
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline


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
    feat_cols = list(eng.columns)
    return eng.values, feat_cols


def train():
    t0 = time.time()
    print("=" * 60)
    print("CROP RECOMMENDATION - FAST V3 TRAINING")
    print("=" * 60)

    X, y, feature_names = load_crop_data()
    X_eng, feat_names = engineer_features(X, feature_names)
    print(f"Features: {X_eng.shape[1]} (engineered from {X.shape[1]})")
    pickle.dump(feat_names, open('models/crop_feature_names.pkl', 'wb'))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_eng)
    pickle.dump(scaler, open('models/crop_scaler.pkl', 'wb'))

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    smote = SMOTE(random_state=42, k_neighbors=3)
    n_classes = len(np.unique(y))

    # Best hyperparameters from prior runs and literature
    configs = {
        'XGBoost': XGBClassifier(
            n_estimators=600, max_depth=9, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, min_child_weight=1,
            gamma=0.1, reg_alpha=0.5, reg_lambda=2.0,
            random_state=42, tree_method='hist', verbosity=0),

        'LGBM': LGBMClassifier(
            n_estimators=600, max_depth=9, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, min_child_weight=1,
            reg_alpha=0.5, reg_lambda=2.0,
            random_state=42, verbose=-1, n_jobs=2),

        'CatBoost': CatBoostClassifier(
            iterations=600, depth=9, learning_rate=0.05,
            l2_leaf_reg=3.0, border_count=128,
            random_seed=42, verbose=0),
    }

    print("\n[1/2] Training and cross-validating base models ...", flush=True)
    scores = {}
    trained = {}
    for name, model in configs.items():
        pipe = IMBPipeline([
            ('smote', smote),
            ('clf', model),
        ])
        cv_scores = cross_val_score(pipe, Xs, y, cv=cv, scoring='accuracy', n_jobs=2)
        scores[name] = cv_scores.mean()
        print(f"  {name:10s}: {cv_scores.mean()*100:.2f}%  (+/- {cv_scores.std()*100:.1f}%)", flush=True)

    # Pick best models for ensemble (take top 2 if all 3 similar)
    best_order = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Best: {best_order[0][0]} ({best_order[0][1]*100:.2f}%)", flush=True)

    # Final split: 70/15/15 (train/val/test)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.15, stratify=y, random_state=42)

    print("\n[2/2] Training final ensemble on train+val ...", flush=True)
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_sm, y_sm = sm.fit_resample(X_tr, y_tr)

    models_to_use = {k: v for k, v in configs.items() if k in [b[0] for b in best_order[:2]]}
    est_list = [(name.lower(), model) for name, model in models_to_use.items()]

    # Soft voting ensemble
    ens = VotingClassifier(estimators=est_list, voting='soft', n_jobs=2)
    ens.fit(X_sm, y_sm)

    y_pred = ens.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1w = f1_score(y_te, y_pred, average='weighted')
    f1m = f1_score(y_te, y_pred, average='macro')

    # Proper CV on full data
    pipe_full = IMBPipeline([('smote', smote), ('ens', ens)])
    cv_preds = cross_val_predict(pipe_full, Xs, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    cv_acc = accuracy_score(y, cv_preds)

    print(f"\n  Ensemble test accuracy : {acc*100:.2f}%")
    print(f"  Weighted F1            : {f1w:.4f}")
    print(f"  Macro F1               : {f1m:.4f}")
    print(f"  5-fold CV accuracy     : {cv_acc*100:.2f}%")
    print(f"  Gap (test - CV)         : {(acc - cv_acc)*100:+.2f}%  (overfit check)", flush=True)

    pickle.dump(ens, open('models/crop_ensemble.pkl', 'wb'))
    print("\n  Saved: models/crop_ensemble.pkl", flush=True)

    print("\nPer-class accuracy (5 worst):")
    for cls in np.argsort([accuracy_score(y_te[y_te==i], y_pred[y_te==i]) if (y_te==i).sum() > 0 else 0
                            for i in range(n_classes)])[:5]:
        mask = y_te == cls
        ca = accuracy_score(y_te[mask], y_pred[mask]) if mask.sum() > 0 else 0
        print(f"  Class {cls:2d}: {ca*100:.1f}% (n={mask.sum()})")

    print("\n" + classification_report(y_te, y_pred, zero_division=0))

    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(cm, display_labels=[f'C{i}' for i in range(n_classes)]).plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.title('Crop Ensemble - Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/crop_confusion_matrix.png', dpi=150)
    print("Saved: outputs/crop_confusion_matrix.png")

    result = {
        'test_accuracy': round(float(acc), 6),
        'weighted_f1': round(float(f1w), 6),
        'macro_f1': round(float(f1m), 6),
        'cv_accuracy': round(float(cv_acc), 6),
        'overfit_gap': round(float(acc - cv_acc), 4),
        'cv_scores': {k: round(float(v), 6) for k, v in scores.items()},
        'n_features': int(X_eng.shape[1]),
        'n_classes': int(n_classes),
    }
    with open('outputs/crop_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: outputs/crop_metrics.json")
    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"** CROP MODEL: {acc*100:.2f}% test | {cv_acc*100:.2f}% CV **")
    print("=" * 60)

if __name__ == '__main__':
    train()
