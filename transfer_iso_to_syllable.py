import warnings
warnings.filterwarnings("ignore")
import glob
import os
import collections
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WLEN, K_WIN, NPCA = 0.5, 2, 20
STOPS = ["b", "p", "d", "t"]

d0 = np.load(FILES[0], allow_pickle=True)
print("=== available fields ===", [c for c in d0.files])
ph_task = pd.DataFrame({c: [str(v) for v in d0[c]] for c in ["task", "phoneme", "place", "manner", "category"]})
sub = ph_task[ph_task.task == "phonemes"]
print("phonemes task: n=%d" % len(sub))
for c in ["phoneme", "place", "manner", "category"]:
    print("  %-9s %s" % (c, dict(collections.Counter(sub[c]).most_common(8))))

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    return X, t, meta

def epoch(X, t, mask, t0, t1, base=(-0.2, -0.11)):
    bm = (t >= base[0]) & (t <= base[1]); wm = (t >= t0) & (t <= t1)
    Xm = X[mask]
    return (Xm - Xm[:, :, bm].mean(2, keepdims=True))[:, :, wm]

def tangent_fit(Xtr):
    pc = PCA(NPCA).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    A = np.stack([pc.transform(z.T).T for z in Xtr], 0)
    T = A.shape[2]; ed = np.linspace(0, T, K_WIN+1).astype(int)
    tss, F = [], []
    for i in range(K_WIN):
        C = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
        ts = TangentSpace(metric="riemann").fit(C)
        tss.append(ts); F.append(ts.transform(C))
    return pc, tss, np.hstack(F)

def tangent_apply(pc, tss, Xte):
    B = np.stack([pc.transform(z.T).T for z in Xte], 0)
    T = B.shape[2]; ed = np.linspace(0, T, K_WIN+1).astype(int)
    return np.hstack([tss[i].transform(Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]]))
                      for i in range(K_WIN)])

OFFSETS = np.arange(-0.1, 0.45, 0.05)
res_within, res_trans = [], []
prof = {o: [] for o in OFFSETS}
print("\n=== TRANSFER: train on ISOLATED phonemes -> test on SYLLABLES ===")
print("  sliding test window (0.5 s) across syllable epoch\n")
for f in FILES:
    X, t, meta = load(f)
    ph = meta.phoneme.values; task = meta.task.values
    ctrl = np.array([s.startswith("control") for s in meta.tms_target.values])
    m_iso = (task == "single-phoneme") & np.isin(ph, STOPS) & ctrl
    m_syl = (task == "phonemes") & np.isin(ph, STOPS) & ctrl
    if m_iso.sum() < 40 or m_syl.sum() < 40:
        print("  %s skip" % os.path.basename(f).split("_")[0]); continue
    Xtr = epoch(X, t, m_iso, 0.0, WLEN)
    ytr = np.where(np.isin(ph[m_iso], ["d", "t"]), 1, 0)
    yte = np.where(np.isin(ph[m_syl], ["d", "t"]), 1, 0)
    pc, tss, Ftr = tangent_fit(Xtr)
    clf = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    clf.fit(Ftr, ytr)
    res_within.append(balanced_accuracy_score(ytr, clf.predict(Ftr)))
    row = []
    for o in OFFSETS:
        Xte = epoch(X, t, m_syl, o, o + WLEN)
        if Xte.shape[2] != Xtr.shape[2]:
            Xte = Xte[:, :, :Xtr.shape[2]] if Xte.shape[2] > Xtr.shape[2] else np.pad(
                Xte, ((0, 0), (0, 0), (0, Xtr.shape[2]-Xte.shape[2])), mode="edge")
        a = balanced_accuracy_score(yte, clf.predict(tangent_apply(pc, tss, Xte)))
        prof[o].append(a); row.append(a)
    res_trans.append(max(row))
    print("  %-8s iso-fit %.3f | transfer profile %s | best %.3f" % (
        os.path.basename(f).split("_")[0], res_within[-1],
        " ".join("%.2f" % v for v in row), max(row)), flush=True)

print("\n=== GROUP: transfer accuracy by test-window offset (chance 0.5) ===")
for o in OFFSETS:
    a = np.array(prof[o]); _, p = ttest_1samp(a, 0.5)
    star = "**" if p < 0.05 and a.mean() > 0.5 else ""
    print("  offset %+.2f s : %.3f +- %.3f  (%d/%d>ch)  p=%.4f %s" % (
        o, a.mean(), a.std(), int((a > 0.5).sum()), len(a), p, star))
