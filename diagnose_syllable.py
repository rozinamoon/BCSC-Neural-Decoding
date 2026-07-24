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
WLEN, K_WIN, NPCA = 0.5, 2, 20
OFFSETS = np.arange(-0.1, 0.35, 0.05)

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    return X, t, meta

def epoch(X, t, mask, t0, t1):
    bm = (t >= -0.2) & (t <= -0.11); wm = (t >= t0) & (t <= t1)
    Xm = X[mask]
    return (Xm - Xm[:, :, bm].mean(2, keepdims=True))[:, :, wm]

def cv_decode(Xw, y, seeds=3):
    accs = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(Xw, y):
            pc = PCA(NPCA).fit(Xw[tr].transpose(0, 2, 1).reshape(-1, Xw.shape[1]))
            prj = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
            A, B = prj(Xw[tr]), prj(Xw[te])
            T = A.shape[2]; ed = np.linspace(0, T, K_WIN+1).astype(int)
            fa, fb = [], []
            for i in range(K_WIN):
                Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
                Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
                ts = TangentSpace(metric="riemann").fit(Ca)
                fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
            c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
            c.fit(np.hstack(fa), y[tr])
            accs.append(balanced_accuracy_score(y[te], c.predict(np.hstack(fb))))
    return float(np.mean(accs))

iso_acc, syl_best = [], []
prof = {o: [] for o in OFFSETS}
print("DIAGNOSTIC: is there ANY decodable place signal inside the syllable task?\n")
for f in FILES:
    X, t, meta = load(f)
    ph = meta.phoneme.values; task = meta.task.values
    ctrl = np.array([s.startswith("control") for s in meta.tms_target.values])
    m_iso = (task == "single-phoneme") & np.isin(ph, STOPS) & ctrl
    m_syl = (task == "phonemes") & np.isin(ph, STOPS) & ctrl
    if m_iso.sum() < 40 or m_syl.sum() < 40:
        print("  %s skip" % os.path.basename(f).split("_")[0]); continue
    y_iso = np.where(np.isin(ph[m_iso], ["d", "t"]), 1, 0)
    y_syl = np.where(np.isin(ph[m_syl], ["d", "t"]), 1, 0)
    a_iso = cv_decode(epoch(X, t, m_iso, 0.0, WLEN), y_iso)
    iso_acc.append(a_iso)
    row = []
    for o in OFFSETS:
        a = cv_decode(epoch(X, t, m_syl, o, o + WLEN), y_syl, seeds=2)
        prof[o].append(a); row.append(a)
    syl_best.append(max(row))
    print("  %-8s n_iso=%d n_syl=%d | ISO %.3f | SYLLABLE by offset: %s | best %.3f" % (
        os.path.basename(f).split("_")[0], m_iso.sum(), m_syl.sum(), a_iso,
        " ".join("%.2f" % v for v in row), max(row)), flush=True)

print("\n=== GROUP ===")
a = np.array(iso_acc); _, p = ttest_1samp(a, 0.5)
print("  ISOLATED (single-phoneme) within-task: %.3f  (%d/%d)  p=%.5f" % (a.mean(), int((a > .5).sum()), len(a), p))
print("\n  SYLLABLE (phonemes task) within-task, by window offset:")
for o in OFFSETS:
    v = np.array(prof[o]); _, p = ttest_1samp(v, 0.5)
    print("    offset %+.2f s : %.3f +- %.3f  (%d/%d>ch)  p=%.4f %s" % (
        o, v.mean(), v.std(), int((v > .5).sum()), len(v), p, "**" if p < 0.05 and v.mean() > .5 else ""))
b = np.array(syl_best); _, p = ttest_1samp(b, 0.5)
print("\n  SYLLABLE best-offset (optimistic, multiple-comparison inflated): %.3f  p=%.5f" % (b.mean(), p))
