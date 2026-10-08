import sys; sys.path.insert(0, 'src')
from crop_features import compute_legit_features, LEGIT_FEATURES
import numpy as np, pickle, os, json, time, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

# Agro-climatic family mapping (46 crops -> 11 families)
FAMILY = {
 'rice':'cereals','wheat':'cereals','maize':'cereals','barley':'cereals',
 'chickpea':'pulses','kidneybeans':'pulses','pigeonpeas':'pulses','mothbeans':'pulses',
 'mungbean':'pulses','blackgram':'pulses','lentil':'pulses','soybean':'pulses','groundnut':'pulses',
 'potato':'vegetables','onion':'vegetables','cabbage':'vegetables','tomato':'vegetables',
 'brinjal':'vegetables','chilli':'vegetables','okra':'vegetables',
 'coriander':'spices','turmeric':'spices','ginger':'spices','cardamom':'spices','blackpepper':'spices',
 'banana':'tropical_fruit','mango':'tropical_fruit','papaya':'tropical_fruit','coconut':'tropical_fruit',
 'apple':'orchard_fruit','orange':'orchard_fruit','grapes':'orchard_fruit','pomegranate':'orchard_fruit',
 'tea':'plantation','coffee':'plantation','rubber':'plantation','cashew':'plantation',
 'cotton':'commercial','jute':'commercial','sugarcane':'commercial','tobacco':'commercial',
 'watermelon':'melons','muskmelon':'melons',
 'sweetpotato':'tubers','tapioca':'tubers',
 'sunflower':'oilseeds','mustard':'oilseeds',
}
BASE15 = ['N','P','K','temperature','humidity','ph','rainfall',
          'N_pct','P_pct','K_pct','N_P_ratio','N_K_ratio','P_K_ratio','hum_temp','rain_ph']


def make_model():
    return XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.75,
        colsample_bytree=0.7, min_child_weight=8, gamma=0.2, reg_alpha=0.5,
        reg_lambda=6.0, max_delta_step=5, random_state=42, tree_method='hist',
        verbosity=0, n_jobs=4)


def train():
    t0 = time.time()
    print("=" * 62)
    print("CROP FAMILY MODEL | 46 crops -> 11 agro-climatic families")
    print("=" * 62, flush=True)

    import pandas as pd
    df = pd.read_csv('data/raw/Crop_recommendation_Extended.csv')
    X_all = compute_legit_features(df['N'], df['P'], df['K'], df['temperature'],
                                   df['humidity'], df['ph'], df['rainfall'])
    idx = [LEGIT_FEATURES.index(f) for f in BASE15]
    X = X_all[:, idx]

    fam_names = sorted(set(FAMILY.values()))
    fam2id = {f: i for i, f in enumerate(fam_names)}
    y = np.array([fam2id[FAMILY[c]] for c in df['label'].values], dtype=np.int32)
    print(f"  X={X.shape}, {len(fam_names)} families", flush=True)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # CV
    print("\n[1/2] 5-fold CV ...", flush=True)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_preds = np.zeros(len(y), dtype=int)
    cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
    sw_map = dict(zip(np.unique(y), np.power(cw, 0.5)))
    for tr, va in skf.split(Xs, y):
        w = np.array([sw_map[c] for c in y[tr]])
        m = make_model(); m.fit(Xs[tr], y[tr], sample_weight=w, verbose=False)
        cv_preds[va] = m.predict(Xs[va])
    cv_acc = accuracy_score(y, cv_preds)
    print(f"    CV top-1: {cv_acc*100:.2f}%", flush=True)

    # Holdout
    print("\n[2/2] Holdout eval ...", flush=True)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.15, stratify=y, random_state=42)
    w_tr = np.array([sw_map[c] for c in y_tr])
    model = make_model()
    model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    proba = model.predict_proba(X_te)
    top3 = (np.argsort(-proba, axis=1)[:, :3] == y_te[:, None]).any(axis=1).mean()
    f1w = f1_score(y_te, y_pred, average='weighted')
    f1m = f1_score(y_te, y_pred, average='macro')

    print(f"\n  ===== FINAL (families, holdout n={len(y_te)}) =====", flush=True)
    print(f"  Top-1 acc : {acc*100:.2f}%", flush=True)
    print(f"  Top-3 acc : {top3*100:.2f}%", flush=True)
    print(f"  Weighted F1: {f1w:.4f}", flush=True)
    print(f"  Macro F1   : {f1m:.4f}", flush=True)
    print(f"  5-fold CV  : {cv_acc*100:.2f}%", flush=True)

    print("\n  Per-family accuracy:", flush=True)
    for c in np.argsort([accuracy_score(y_te[y_te==i], y_pred[y_te==i]) if (y_te==i).sum()>0 else 0 for i in range(len(fam_names))]):
        mask = y_te == c
        ca = accuracy_score(y_te[mask], y_pred[mask]) if mask.sum()>0 else 0
        print(f"    {fam_names[c]:16s}: {ca*100:5.1f}% (n={mask.sum()})", flush=True)

    # Save
    pickle.dump(model, open('models/xgb_crop_family.pkl', 'wb'))
    pickle.dump(scaler, open('models/crop_family_scaler.pkl', 'wb'))
    pickle.dump({'family_names': fam_names, 'fam2id': fam2id,
                 'crop_to_family': FAMILY, 'features': BASE15},
                open('models/crop_family_meta.pkl', 'wb'))
    result = {
        'test_top1': round(float(acc), 6), 'test_top3': round(float(top3), 6),
        'weighted_f1': round(float(f1w), 6), 'macro_f1': round(float(f1m), 6),
        'cv_top1': round(float(cv_acc), 6), 'n_families': len(fam_names),
        'families': fam_names, 'features': BASE15, 'leak_free': True,
        'note': 'Family-level crop model. Individual-crop v11: 76.85% top-1 / 97.75% top-3.',
    }
    with open('outputs/crop_family_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: models/xgb_crop_family.pkl + outputs/crop_family_metrics.json ({time.time()-t0:.0f}s)", flush=True)
    print("=" * 62)


if __name__ == '__main__':
    train()
