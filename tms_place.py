import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score

FILES = [f for f in sorted(glob.glob("data/npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
BMIN, BMAX = -0.2, -0.11
WMIN, WMAX = 0.0, 0.5
volt = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))

def acc(F: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        return np.nan
    return float(np.mean([cross_val_score(volt, F, y, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                                          scoring="balanced_accuracy").mean() for s in range(5)]))

CONDS = {"control": ["control_lip", "control_tongue"], "lip": ["lip"], "tongue": ["tongue"]}
R = {c: [] for c in CONDS}
print("TMS x place decoding | phonemes task, S* subjects")
print("baseline %.2f-%.2f (pre-TMS), decode %.1f-%.1fs | place=alveolar-vs-bilabial (voltage-LDA)\n" % (BMIN, BMAX, WMIN, WMAX))
print("%-8s %10s %10s %10s" % ("subj", "control", "lip", "tongue"))
for f in FILES:
    sub = os.path.basename(f).split("_")[0]
    d = np.load(f, allow_pickle=True); X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({k: d[k] for k in ["task", "tms_target", "place"]})
    ph = (df.task == "phonemes").values
    bmask = (t >= BMIN) & (t <= BMAX); wmask = (t >= WMIN) & (t <= WMAX)
    row = []
    for cond, tgts in CONDS.items():
        cm = ph & df.tms_target.isin(tgts).values & np.isin(df.place.values, ["alveolar", "bilabial"])
        Xc = X[cm]
        Xb = (Xc - Xc[:, :, bmask].mean(2, keepdims=True))[:, :, wmask]
        F = Xb[:, :, ::10].reshape(len(Xb), -1)
        y = (df.place.values[cm] == "alveolar").astype(int)
        a = acc(F, y)
        R[cond].append(a); row.append(a)
    print("%-8s %10.3f %10.3f %10.3f" % (sub, row[0], row[1], row[2]), flush=True)

print("\n=== GROUP (place decoding accuracy per TMS condition, chance 0.5) ===")
for c in CONDS:
    a = np.array(R[c]); a = a[~np.isnan(a)]
    _, p = ttest_1samp(a, 0.5)
    print("  %-8s mean %.3f +- %.3f  (%d/%d>chance)  vs-chance p=%.4f" % (c, a.mean(), a.std(), int((a > 0.5).sum()), len(a), p))
print("\n=== MODULATION (paired-t across conditions) ===")
ctrl = np.array(R["control"]); lip = np.array(R["lip"]); ton = np.array(R["tongue"])
ok = ~(np.isnan(ctrl) | np.isnan(lip) | np.isnan(ton))
ctrl, lip, ton = ctrl[ok], lip[ok], ton[ok]
for n1, a1, n2, a2 in [("lip", lip, "control", ctrl), ("tongue", ton, "control", ctrl), ("lip", lip, "tongue", ton)]:
    _, p = ttest_rel(a1, a2)
    print("  %s vs %s: %+.3f  paired-p=%.4f %s" % (n1, n2, a1.mean()-a2.mean(), p, "*" if p < 0.05 else ""))
