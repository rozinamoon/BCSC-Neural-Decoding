import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score

FILES = [f for f in sorted(glob.glob("data/npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
BMIN, BMAX = -0.2, -0.11
K = 20

def load(f: str):
    d = np.load(f, allow_pickle=True); X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme", "place"]})
    bm = (t >= BMIN) & (t <= BMAX); wm = (t >= WMIN) & (t <= WMAX)
    def feat(mask):
        Xm = X[mask]; return (Xm - Xm[:, :, bm].mean(2, keepdims=True))[:, :, wm]
    ph = df.phoneme.values
    ctrl = ((df.task == "single-phoneme") & df.tms_target.str.startswith("control")).values
    tms = (df.task == "phonemes") & df.tms_target.isin(["lip", "tongue"]).values
    tctrl = (df.task == "phonemes") & df.tms_target.isin(["control_lip", "control_tongue"]).values
    Xc = feat(ctrl); phc = ph[ctrl]
    localizer = np.abs(feat(tms).mean(0).mean(1) - feat(tctrl).mean(0).mean(1))
    return Xc, phc, localizer

subj = []
for f in FILES:
    Xc, phc, loc = load(f)
    sm = np.isin(phc, ["b", "p", "d", "t"])
    if sm.sum() >= 20 and min(np.bincount(np.isin(phc[sm], ["d", "t"]).astype(int))) >= 8:
        y = np.where(np.isin(phc[sm], ["d", "t"]), 1, 0)
        subj.append((os.path.basename(f).split("_")[0], Xc[sm], y, loc))

def decode_within(chans):
    accs = []
    for _, Xb, y, loc in subj:
        ch = chans(loc)
        F = Xb[:, ch, ::10].reshape(len(Xb), -1)
        clf = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
        accs.append(np.mean([cross_val_score(clf, F, y, cv=StratifiedKFold(5, shuffle=True, random_state=s), scoring="balanced_accuracy").mean() for s in range(3)]))
    return np.array(accs)

def decode_loso(group_chans: bool):
    accs = []
    for i in range(len(subj)):
        idx = [j for j in range(len(subj)) if j != i]
        if group_chans:
            gloc = np.mean([subj[j][3] for j in idx], 0)
            ch = np.argsort(gloc)[-K:]
        else:
            ch = np.arange(61)
        Ftr = np.concatenate([subj[j][1][:, ch, ::10].reshape(len(subj[j][1]), -1) for j in idx], 0)
        ytr = np.concatenate([subj[j][2] for j in idx], 0)
        Fte = subj[i][1][:, ch, ::10].reshape(len(subj[i][1]), -1)
        clf = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
        clf.fit(Ftr, ytr)
        accs.append(balanced_accuracy_score(subj[i][2], clf.predict(Fte)))
    return np.array(accs)

rng = np.random.default_rng(0)
sels = {"all-61": lambda l: np.arange(61),
        "TMS-top20": lambda l: np.argsort(l)[-K:],
        "TMS-bottom20": lambda l: np.argsort(l)[:K],
        "random20": lambda l: rng.choice(61, K, replace=False)}
print("PLACE-STOPS | %d subj | TMS-localizer channel selection (leakage-safe: localizer uses TMS/control not place)\n" % len(subj))
print("WITHIN-subject:")
for name, fn in sels.items():
    a = decode_within(fn); _, p = ttest_1samp(a, 0.5)
    print("  %-13s %.3f +- %.3f  (%d/%d)  p=%.4f %s" % (name, a.mean(), a.std(), int((a > 0.5).sum()), len(a), p, "*" if p < 0.05 and a.mean() > 0.5 else ""))
print("\nCROSS-subject LOSO:")
for name, gc in [("all-61", False), ("TMS-top20 (group)", True)]:
    a = decode_loso(gc); _, p = ttest_1samp(a, 0.5)
    print("  %-18s %.3f +- %.3f  (%d/%d)  p=%.4f %s" % (name, a.mean(), a.std(), int((a > 0.5).sum()), len(a), p, "**SIG**" if p < 0.05 and a.mean() > 0.5 else ""))
