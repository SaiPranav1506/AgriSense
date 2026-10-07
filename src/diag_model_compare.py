import sys; sys.path.insert(0, 'src')
from crop_features import compute_legit_features, LEGIT_FEATURES
import numpy as np, pickle, time, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

df = pd.read_csv('data/raw/Crop_recommendation_Extended.csv')
le = pickle.load(open('models/label_encoder.pkl','rb'))
y = le.transform(df['label'].values).astype(int)
BASE15 = ['N','P','K','temperature','humidity','ph','rainfall',
          'N_pct','P_pct','K_pct','N_P_ratio','N_K_ratio','P_K_ratio','hum_temp','rain_ph']
X_all = compute_legit_features(df['N'],df['P'],df['K'],df['temperature'],df['humidity'],df['ph'],df['rainfall'])
idx = [LEGIT_FEATURES.index(f) for f in BASE15]
X = StandardScaler().fit_transform(X_all[:, idx])
print(f"Data: {X.shape}, {len(np.unique(y))} classes, base15 leak-free features\n")

sw = compute_class_weight('balanced', classes=np.unique(y), y=y)
sw_map = dict(zip(np.unique(y), np.power(sw, 0.5)))
def wfor(idx_): return np.array([sw_map[c] for c in y[idx_]])

def run_cv(make, tag, needs_w=True):
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    preds = np.zeros(len(y), dtype=int); t0=time.time()
    for tr,va in skf.split(X,y):
        m = make()
        if needs_w:
            m.fit(X[tr], y[tr], sample_weight=wfor(tr))
        else:
            m.fit(X[tr], y[tr])
        preds[va] = np.asarray(m.predict(X[va])).ravel()
    acc = accuracy_score(y, preds)
    print(f"  {tag:34s}: {acc*100:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    return acc, preds

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

results = {}
results['xgb_current'] = run_cv(lambda: XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.75,
    colsample_bytree=0.7, min_child_weight=8, gamma=0.2, reg_alpha=0.5,
    reg_lambda=6.0, max_delta_step=5, random_state=42, tree_method='hist',
    n_jobs=4, verbosity=0), 'XGBoost (current v11)')

results['lgbm'] = run_cv(lambda: LGBMClassifier(
    n_estimators=600, max_depth=-1, num_leaves=63, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    reg_alpha=0.3, reg_lambda=3.0, random_state=42, n_jobs=4, verbose=-1),
    'LightGBM (tuned)', needs_w=False)

results['catboost'] = run_cv(lambda: CatBoostClassifier(
    iterations=800, depth=7, learning_rate=0.05, l2_leaf_reg=5.0,
    random_state=42, verbose=0, thread_count=4), 'CatBoost (tuned)', needs_w=False)

# XGBoost tuned harder
results['xgb_tuned'] = run_cv(lambda: XGBClassifier(
    n_estimators=900, max_depth=8, learning_rate=0.05, subsample=0.85,
    colsample_bytree=0.8, min_child_weight=4, gamma=0.1, reg_alpha=0.2,
    reg_lambda=3.0, random_state=42, tree_method='hist', n_jobs=4, verbosity=0),
    'XGBoost (tuned)')

print("\n=== Summary (5-fold CV, leak-free) ===")
for k,v in sorted(results.items(), key=lambda x:-x[1][0]):
    print(f"  {k:16s}: {v[0]*100:.2f}%")

# Save preds for ensemble check
np.save('outputs/_cmp_xgb.npy', results['xgb_current'][1])
np.save('outputs/_cmp_lgbm.npy', results['lgbm'][1])
np.save('outputs/_cmp_cb.npy', results['catboost'][1])
np.save('outputs/_cmp_xgbt.npy', results['xgb_tuned'][1])
