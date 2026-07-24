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
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
STOPS = ["b", "p", "d", "t"]
NPCA = 20
WINDOWS = [(0.0, 0.10), (0.03, 0.13), (0.05, 0.15), (0.0, 0.15), (0.05, 0.20),
           (0.10, 0.25), (0.0, 0.25), (0.0, 0.50)]

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    return X, t, meta

def cut(X, t, mask, t0, t1):
    bm = (t >= -0.2) & (t <= -0.11); wm = (t >= t0) & (t <= t1)
    Xm = X[mask]
    return (Xm - Xm[:, :, bm].mean(2, keepdims=True))[:, :, wm]

def featurise(Xtr, Xte, K):
    pc = PCA(min(NPCA, len(Xtr)-2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    prj = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = prj(Xtr), prj(Xte)
    T = A.shape[2]; K = max(1, min(K, T // 12)); ed = np.linspace(0, T, K+1).astype(int)
    fa, fb = [], []
    for i in range(K):
        Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
        Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
    return np.hstack(fa), np.hstack(fb)

def cvd(Xw, y, K=2, seeds=3):
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 6: return np.nan
    acc = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=s).split(Xw, y):
            a, b = featurise(Xw[tr], Xw[te], K)
            c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
            c.fit(a, y[tr]); acc.append(balanced_accuracy_score(y[te], c.predict(b)))
    return float(np.mean(acc))

R2 = {w: [] for w in WINDOWS}
R4 = {w: [] for w in WINDOWS}
ISO2 = {w: [] for w in WINDOWS}
print("SHORT WINDOWS matched to consonant timescale | syllable (CV/VC) place & 4-way\n")
for f in FILES:
    X, t, meta = load(f)
    ph = meta.phoneme.values; task = meta.task.values
    ctrl = np.array([s.startswith("control") for s in meta.tms_target.values])
    m_syl = (task == "phonemes") & np.isin(ph, STOPS) & ctrl
    m_iso = (task == "single-phoneme") & np.isin(ph, STOPS) & ctrl
    if m_syl.sum() < 40 or m_iso.sum() < 40: continue
    y2s = np.where(np.isin(ph[m_syl], ["d", "t"]), 1, 0)
    y4s = np.array([STOPS.index(v) for v in ph[m_syl]])
    y2i = np.where(np.isin(ph[m_iso], ["d", "t"]), 1, 0)
    row = []
    for w in WINDOWS:
        Xs = cut(X, t, m_syl, *w); Xi = cut(X, t, m_iso, *w)
        R2[w].append(cvd(Xs, y2s)); R4[w].append(cvd(Xs, y4s)); ISO2[w].append(cvd(Xi, y2i))
        row.append("%.2f/%.2f" % (R2[w][-1], ISO2[w][-1]))
    print("  %-8s syl/iso place by window: %s" % (os.path.basename(f).split("_")[0], " ".join(row)), flush=True)

print("\n=== SYLLABLE (CV/VC) — place binary, chance 0.5 ===")
for w in WINDOWS:
    a = np.array([v for v in R2[w] if not np.isnan(v)]); _, p = ttest_1samp(a, 0.5)
    print("  win %.2f-%.2f s : %.3f +- %.3f  (%d/%d)  p=%.4f %s" % (
        w[0], w[1], a.mean(), a.std(), int((a > .5).sum()), len(a), p, "**" if p < 0.05 and a.mean() > .5 else ""))
print("\n=== SYLLABLE — 4-way consonant identity, chance 0.25 ===")
for w in WINDOWS:
    a = np.array([v for v in R4[w] if not np.isnan(v)]); _, p = ttest_1samp(a, 0.25)
    print("  win %.2f-%.2f s : %.3f +- %.3f  (%d/%d)  p=%.4f %s" % (
        w[0], w[1], a.mean(), a.std(), int((a > .25).sum()), len(a), p, "**" if p < 0.05 and a.mean() > .25 else ""))
print("\n=== ISOLATED — place binary (reference), chance 0.5 ===")
for w in WINDOWS:
    a = np.array([v for v in ISO2[w] if not np.isnan(v)]); _, p = ttest_1samp(a, 0.5)
    print("  win %.2f-%.2f s : %.3f +- %.3f  (%d/%d)  p=%.4f %s" % (
        w[0], w[1], a.mean(), a.std(), int((a > .5).sum()), len(a), p, "**" if p < 0.05 and a.mean() > .5 else ""))
