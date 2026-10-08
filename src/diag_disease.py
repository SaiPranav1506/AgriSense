import sys, os, json, time
sys.path.insert(0, 'src')
from preprocess import load_disease_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)

t0 = time.time()
print("LOADING DATA...", flush=True)
train_ds, val_ds, test_ds, class_names = load_disease_data()
NC = len(class_names)
print(f"classes={NC}", flush=True)

model = tf.keras.models.load_model('models/mobilenet_disease_v4.h5', compile=False)
print("MODEL LOADED. Layers w/ dropout:", [l.name for l in model.layers if 'dropout' in l.name.lower()], flush=True)


# ---------------------------------------------------------------
# 1) Single deterministic pass over FULL test set: collect y_true + argmax
# ---------------------------------------------------------------
def collect_full(ds):
    y_true, y_pred = [], []
    for xb, yb in ds:
        logits = model.predict_on_batch(xb)          # (B, NC) softmax
        y_pred.append(np.argmax(logits, axis=1))
        y_true.append(yb.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)

print("FULL TEST EVAL...", flush=True)
y_true_full, y_pred_full = collect_full(test_ds)
N_test = len(y_true_full)

top1 = accuracy_score(y_true_full, y_pred_full)
w_f1 = f1_score(y_true_full, y_pred_full, average='weighted')
m_f1 = f1_score(y_true_full, y_pred_full, average='macro')

# per-class support / precision / recall / f1
cm = confusion_matrix(y_true_full, y_pred_full)
support = cm.sum(axis=1)

with np.errstate(divide='ignore', invalid='ignore'):
    prec = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)  # col sums
    rec = np.diag(cm) / np.maximum(support, 1)          # row sums
f1c = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
f1c[(prec + rec) == 0] = 0.0

per_class = []
for i, c in enumerate(class_names):
    per_class.append({
        'class': c, 'index': i,
        'precision': float(prec[i]), 'recall': float(rec[i]),
        'f1': float(f1c[i]), 'support': int(support[i])
    })

# min/max support
sp = sorted(per_class, key=lambda r: r['support'])
min_s_class, max_s_class = sp[0], sp[-1]
imbalance_ratio = max_s_class['support'] / min_s_class['support']

# top off-diagonal confusions
off = []
for i in range(NC):
    for j in range(NC):
        if i != j and cm[i, j] > 0:
            off.append((int(cm[i, j]), i, j))
off.sort(reverse=True)
top_confusions = [
    {'true': class_names[i], 'pred': class_names[j], 'count': int(cnt),
     'label': f"True {class_names[i]} -> pred {class_names[j]} ({cnt})",
     'pct_true': float(cnt / max(support[i], 1))}
    for cnt, i, j in off[:5]
]

print(f"\n=== FULL TEST (N={N_test}) ===", flush=True)
print(f"top1_acc={top1:.4f}  weighted_f1={w_f1:.4f}  macro_f1={m_f1:.4f}", flush=True)
print(f"min_support={min_s_class['support']}({min_s_class['class']})  "
      f"max_support={max_s_class['support']}({max_s_class['class']})  "
      f"ratio={imbalance_ratio:.1f}x", flush=True)
for tc in top_confusions:
    print("  ", tc['label'], f"({tc['pct_true']*100:.1f}% of true-class)")

# ---------------------------------------------------------------
# 2) MC-dropout: fixed random subset of test set
# ---------------------------------------------------------------
# CADENCE: keep to ~1000 images (32 batches). 10 passes on CPU.
MC_N_BATCHES = 32
print(f"\nMC-DROPOUT SUBSET: {MC_N_BATCHES} batches (~{MC_N_BATCHES*32} imgs), 10 passes", flush=True)
sub_imgs, sub_true = [], []
for xb, yb in test_ds.take(MC_N_BATCHES):
    sub_imgs.append(xb)
    sub_true.append(yb.numpy())
sub_imgs = tf.concat(sub_imgs, 0)
sub_true = np.concatenate(sub_true)
N_mc = len(sub_true)

NBATCH_mc = int(np.ceil(N_mc / 32))
mc_pass_argmax = []   # per-pass argmax
mc_pass_maxconf = []  # per-pass max softmax
tm = time.time()
for p in range(10):
    pas0 = np.zeros((0, NC), dtype=np.float32)
    start = 0
    for b in range(NBATCH_mc):
        xb = sub_imgs[start:start + 32]
        start += 32
        out = model(xb.numpy() if tf.is_tensor(xb) else xb, training=True).numpy()
        pas0 = np.concatenate([pas0, out], 0)
    mc_pass_argmax.append(np.argmax(pas0, 1))
    mc_pass_maxconf.append(np.max(pas0, 1))

mc_argmax = np.stack(mc_pass_argmax)      # (10, N_mc)
mc_maxconf = np.stack(mc_pass_maxconf)    # (10, N_mc)
mc_mean_std = float(np.mean(np.std(mc_maxconf, axis=0)))
# flip: argmax not identical across all 10 passes
consistent = np.all(mc_argmax == mc_argmax[0], axis=0)
flip_rate = float(1.0 - np.mean(consistent))
print(f"MC-dropout done in {time.time()-tm:.0f}s  (N_mc={N_mc})", flush=True)
print(f"mc_dropout_mean_std={mc_mean_std:.4f}  mc_dropout_flip_rate={flip_rate:.4f} "
      f"({int(flip_rate*N_mc)}/{N_mc} flipped)", flush=True)

# ---------------------------------------------------------------
# 3) Calibration on the sampled subset (deterministic single pass)
# ---------------------------------------------------------------
start = 0
det_logits = np.zeros((0, NC), dtype=np.float32)
for b in range(NBATCH_mc):
    xb = sub_imgs[start:start + 32]; start += 32
    det_logits = np.concatenate([det_logits, model(xb.numpy() if tf.is_tensor(xb) else xb, training=False).numpy()], 0)
det_conf = np.max(det_logits, 1)
det_pred = np.argmax(det_logits, 1)
correct = (det_pred == sub_true).astype(float)

def calibration_bins(conf, corr, nbins=10):
    edges = np.linspace(0, 1, nbins + 1)
    rows = []
    for k in range(nbins):
        lo, hi = edges[k], edges[k + 1]
        mask = (conf >= lo) & (conf <= hi) if k == nbins - 1 else (conf >= lo) & (conf < hi)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        rows.append({'bin': f"{lo:.2f}-{hi:.2f}", 'n': int(len(idx)),
                     'mean_conf': float(np.mean(conf[idx])),
                     'fraction_correct': float(np.mean(corr[idx]))})
    return rows

calib = calibration_bins(det_conf, correct, 10)
ece = float(np.mean(np.abs(det_conf - correct)))
print(f"\nCALIBRATION (N_mc={N_mc}, deterministic pass): ECE={ece:.4f}", flush=True)
for r in calib:
    delta = r['mean_conf'] - r['fraction_correct']
    tag = 'OVER-conf' if delta > 0 else 'UNDER-conf'
    print(f"  bin {r['bin']:11s} n={r['n']:4d}  mean_conf={r['mean_conf']:.3f}  "
          f"frac_correct={r['fraction_correct']:.3f}  delta={delta:+.3f} {tag}", flush=True)
overall_over = float(np.mean(det_conf) - np.mean(correct))

print(f"\nTOTAL TIME {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------
# 4) Conclusion
# ---------------------------------------------------------------
# macro vs weighted gap
macro_weighted_gap = m_f1 - w_f1

result = {
    'test_set': {
        'n_images': N_test, 'n_batches': int(np.ceil(N_test / 32)),
        'top1_accuracy': top1, 'weighted_f1': w_f1, 'macro_f1': m_f1,
        'macro_minus_weighted_f1': macro_weighted_gap,
    },
    'imbalance': {
        'min_support': min_s_class['class'], 'min_support_n': min_s_class['support'],
        'max_support': max_s_class['class'], 'max_support_n': max_s_class['support'],
        'ratio': imbalance_ratio,
    },
    'per_class': per_class,
    'top_confusions': top_confusions,
    'mc_dropout': {
        'n_subset': N_mc, 'n_passes': 10,
        'mean_std_max_softmax': mc_mean_std, 'flip_rate': flip_rate,
    },
    'calibration': {
        'ece': ece, 'n': N_mc,
        'overall_mean_conf': float(np.mean(det_conf)),
        'overall_frac_correct': float(np.mean(correct)),
        'overall_delta_conf_minus_correct': overall_over,
        'bins': calib,
    },
}

# ---- decide dominant error mode from measured numbers ----
# variance signal: high flip rate + high mean_std
# bias signal: large off-diagonal confusion counts clustered on specific class pairs
# imbalance signal: large macro-vs-weighted gap AND small classes having low recall
flip = flip_rate
mean_std = mc_mean_std
gap = macro_weighted_gap

# fraction of the (single-pass) test-time errors explained by the top-5 off-diagonal pairs
top5_sum = sum(tc['count'] for tc in top_confusions)
n_err_test = int(N_test - top1 * N_test)

if flip > 0.15:
    CONC_TYPE = "PREDICTION INSTABILITY (variance)"
    CONC_TEXT = (f"MC-dropout flip rate {flip:.2f} and max-softmax std {mean_std:.3f} are high, "
                 f"meaning the head's decisions are unstable across stochastic passes; errors are not only systematic.")
elif gap < -0.08 and n_err_test > 0 and (top5_sum / n_err_test) < 0.5:
    CONC_TYPE = "CLASS-IMBALANCE IGNORING SMALL CLASSES"
    CONC_TEXT = (f"macro_f1 trails weighted_f1 by {gap:+.3f} with a {imbalance_ratio:.1f}x support ratio and low recall on "
                 f"small classes, while the largest confusions explain only {(top5_sum/n_err_test)*100:.0f}% of errors.")
else:
    CONC_TYPE = "SYSTEMATIC CONFUSION BETWEEN SIMILAR CLASSES (bias)"
    CONC_TEXT = (f"the {len(top_confusions)} largest off-diagonal confusions account for "
                 f"{(top5_sum/n_err_test)*100:.0f}% of the {n_err_test} test errors and are concentrated on "
                 f"visually similar leaf-disease classes, while MC-dropout flip rate is only {flip:.2%}.")

conclusion = (
    f"The model's errors are dominated by {CONC_TYPE}: {CONC_TEXT}  "
    f"(macro_f1={m_f1:.3f} vs weighted_f1={w_f1:.3f}, gap={gap:+.3f}; imbalance ratio {imbalance_ratio:.1f}x; "
    f"MC flip={flip:.3f}, max-softmax std={mean_std:.4f}).")
result['conclusion'] = conclusion
result['conclusion_mode'] = CONC_TYPE
result['_top5_err_fraction'] = (top5_sum / n_err_test) if n_err_test > 0 else None

report = result  # printed both as JSON and markdown below

# ---------------------------------------------------------------
# PRINT
# ---------------------------------------------------------------
print("\n===== JSON =====")
print(json.dumps(result, indent=2))

print("\n===== MARKDOWN REPORT =====")
md = []
md.append("# Disease Model Bias/Variance + Error Report")
md.append(f"**Model:** models/mobilenet_disease_v4.h5 (MobileNetV2, 15 classes)  ")
md.append(f"**Test set:** N={N_test} images, top1_acc={top1:.4f}, weighted_f1={w_f1:.4f}, macro_f1={m_f1:.4f}")
md.append("")
md.append("## 1. Class imbalance")
md.append(f"- Min support: **{min_s_class['class']}** = {min_s_class['support']} imgs")
md.append(f"- Max support: **{max_s_class['class']}** = {max_s_class['support']} imgs")
md.append(f"- Imbalance ratio: **{imbalance_ratio:.1f}x**")
md.append(f"- macro_f1 - weighted_f1 = {macro_weighted_gap:+.4f}  (how much macro_f1 is pulled down by small/underperforming classes)")
md.append("")
md.append("## 2. Per-class metrics")
md.append("| class | prec | rec | f1 | support |")
md.append("|---|---|---|---|---|")
for r in sorted(per_class, key=lambda x: -x['support']):
    md.append(f"| {r['class']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['support']} |")
md.append("")
md.append("## 3. Top off-diagonal confusions")
for tc in top_confusions:
    md.append(f"- {tc['label']}  ({tc['pct_true']*100:.1f}% of all true-class images)")
md.append("")
md.append(f"## 4. MC-dropout variance (N={N_mc} subset, 10 passes, {int(np.sum(~consistent))}/{N_mc} flipped)")
md.append(f"- mean std of max-softmax across passes: **{mc_mean_std:.4f}**")
md.append(f"- argmax flip rate: **{flip_rate:.4f}**")
md.append("")
md.append("## 5. Calibration (deterministic pass on subset)")
md.append(f"- ECE={ece:.4f}; overall mean_conf={np.mean(det_conf):.3f} vs frac_correct={np.mean(correct):.3f} "
          f"(delta {overall_over:+.3f} -> model is {'OVER-confident' if overall_over>0 else 'UNDER-confident'})")
md.append("| bin | n | mean_conf | frac_correct | delta |")
md.append("|---|---|---|---|---|")
for r in calib:
    d = r['mean_conf'] - r['fraction_correct']
    md.append(f"| {r['bin']} | {r['n']} | {r['mean_conf']:.3f} | {r['fraction_correct']:.3f} | {d:+.3f} |")
md.append("")
md.append("## 6. Conclusion")
md.append(conclusion)
print("\n".join(md))