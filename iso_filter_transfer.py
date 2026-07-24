import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.spatialfilters import Xdawn

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
STOPS = ["b", "p", "d", "t"]
NF, K_WIN = 6, 2

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    return X, t, meta

def cut(X, t, mask, t0=0.0, t1=0.5):
    bm = (t >= -0.2) & (t <= -0.11); wm = (t >= t0) & (t <= t1)
    Xm = X[mask]
    return (Xm - Xm[:, :, bm].mean(2, keepdims=True))[:, :, wm]

def tan_feats(Atr, Ate, K=K_WIN, npca=None):
    if npca:
        pc = PCA(min(npca, Atr.shape[1])).fit(Atr.transpose(0, 2, 1).reshape(-1, Atr.shape[1]))
        prj = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
        Atr, Ate = prj(Atr), prj(Ate)
    T = Atr.shape[2]; ed = np.linspace(0, T, K+1).astype(int)
    fa, fb = [], []
    for i in range(K):
        Ca = Covariances("oas").fit_transform(Atr[:, :, ed[i]:ed[i+1]])
        Cb = Covariances("oas").fit_transform(Ate[:, :, ed[i]:ed[i+1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
    return np.hstack(fa), np.hstack(fb)

def cvd(Xw, y, npca=20, seeds=3):
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 6: return np.nan
    acc = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=s).split(Xw, y):
            a, b = tan_feats(Xw[tr], Xw[te], npca=npca)
            c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
            c.fit(a, y[tr]); acc.append(balanced_accuracy_score(y[te], c.predict(b)))
    return float(np.mean(acc))

R = {k: [] for k in ["syl_raw", "SYL_ISOFILTER", "syl_ownfilter", "iso_ref", "syl_isofilt_4way"]}
print("Does an ISOLATED-derived spatial filter expose the consonant inside a syllable?\n")
for f in FILES:
    X, t, meta = load(f)
    ph = meta.phoneme.values; task = meta.task.values
    ctrl = np.array([s.startswith("control") for s in meta.tms_target.values])
    m_iso = (task == "single-phoneme") & np.isin(ph, STOPS) & ctrl
    m_syl = (task == "phonemes") & np.isin(ph, STOPS) & ctrl
    if m_iso.sum() < 40 or m_syl.sum() < 40: continue
    Xi, Xs = cut(X, t, m_iso), cut(X, t, m_syl)
    yi = np.where(np.isin(ph[m_iso], ["d", "t"]), 1, 0)
    ys = np.where(np.isin(ph[m_syl], ["d", "t"]), 1, 0)
    ys4 = np.array([STOPS.index(v) for v in ph[m_syl]])
    R["iso_ref"].append(cvd(Xi, yi))
    R["syl_raw"].append(cvd(Xs, ys))
    xd_iso = Xdawn(nfilter=NF, estimator="oas").fit(Xi, yi)
    Zs = xd_iso.transform(Xs)
    R["SYL_ISOFILTER"].append(cvd(Zs, ys, npca=None))
    R["syl_isofilt_4way"].append(cvd(Zs, ys4, npca=None))
    own = []
    for s in range(3):
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=s).split(Xs, ys):
            xd = Xdawn(nfilter=NF, estimator="oas").fit(Xs[tr], ys[tr])
            a, b = tan_feats(xd.transform(Xs[tr]), xd.transform(Xs[te]))
            c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
            c.fit(a, ys[tr]); own.append(balanced_accuracy_score(ys[te], c.predict(b)))
    R["syl_ownfilter"].append(float(np.mean(own)))
    print("  %-8s iso %.3f | syl raw %.3f | syl+ISO-filter %.3f | syl own-filter %.3f | 4way %.3f" % (
        os.path.basename(f).split("_")[0], R["iso_ref"][-1], R["syl_raw"][-1],
        R["SYL_ISOFILTER"][-1], R["syl_ownfilter"][-1], R["syl_isofilt_4way"][-1]), flush=True)

print("\n=== GROUP ===")
for k, ch, lbl in [("iso_ref", .5, "ISOLATED place (reference)"),
                   ("syl_raw", .5, "syllable place, no filter"),
                   ("SYL_ISOFILTER", .5, "syllable place, ISOLATED-derived filter"),
                   ("syl_ownfilter", .5, "syllable place, own Xdawn filter"),
                   ("syl_isofilt_4way", .25, "syllable 4-way, ISO filter")]:
    a = np.array([v for v in R[k] if not np.isnan(v)]); _, p = ttest_1samp(a, ch)
    print("  %-42s %.3f +- %.3f  (%d/%d)  p=%.5f %s" % (
        lbl, a.mean(), a.std(), int((a > ch).sum()), len(a), p, "**" if p < 0.05 and a.mean() > ch else ""))
_, pc = ttest_rel(R["SYL_ISOFILTER"], R["syl_raw"])
print("\n  ISO-filter vs raw syllable: %+.3f (p=%.4f)" % (
    np.mean(R["SYL_ISOFILTER"]) - np.mean(R["syl_raw"]), pc))
