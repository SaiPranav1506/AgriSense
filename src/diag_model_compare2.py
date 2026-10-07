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
print(f"Data: {X.shape}, {len(np.unique(y))} classes", flush=True)

sw = compute_class_weight('balanced', classes=np.unique(y), y=y)
sw_map = dict(zip(np.unique(y), np.power(sw, 0.5)))
def wfor(ii): return np.array([sw_map[c] for c in y[ii]])

def run_cv(make, tag, needs_w=True):
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    preds = np.zeros(len(y), dtype=int); t0=time.time()
    for tr,va in skf.split(X,y):
        m = make()
        m.fit(X[tr], y[tr], sample_weight=wfor(tr)) if needs_w else m.fit(X[tr], y[tr])
        preds[va] = np.asarray(m.predict(X[va])).ravel()
    acc = accuracy_score(y, preds)
    print(f"  {tag:30s}: {acc*100:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    return acc

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

print("Running CatBoost (400 iters) + XGBoost tuned...", flush=True)
cb = run_cv(lambda: CatBoostClassifier(iterations=400, depth=7, learning_rate=0.08,
    l2_leaf_reg=5.0, random_state=42, verbose=0, thread_count=4), 'CatBoost (400 iters)', needs_w=False)
xgbt = run_cv(lambda: XGBClassifier(n_estimators=900, max_depth=8, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.8, min_child_weight=4, gamma=0.1,
    reg_alpha=0.2, reg_lambda=3.0, random_state=42, tree_method='hist', n_jobs=4, verbosity=0),
    'XGBoost (tuned)')

print(f"\nRESULTS: CatBoost={cb*100:.2f}%  XGBoost_tuned={xgbt*100:.2f}%  (baseline XGB=76.23%)", flush=True)
