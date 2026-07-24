import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space as tsf

PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
WIN, NPCA_COV, P_STD = 50, 10, 0.3


def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    k = ((meta.task == "single-phoneme") & meta.tms_target.str.startswith("control")).values
    bm = (t >= -0.2) & (t <= -0.11); wm = (t >= 0.0) & (t <= 0.5)
    Xk = X[k]
    return (Xk - Xk[:, :, bm].mean(2, keepdims=True))[:, :, wm], meta.phoneme.values[k]


def obs_pc1(X):
    pc = PCA(1).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
    return np.stack([pc.transform(z.T) for z in X], 0)[:, :, 0]


def obs_geom(X):
    pc = PCA(NPCA_COV).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
    Z = np.stack([pc.transform(z.T).T for z in X], 0)
    T = Z.shape[2]
    C = np.stack([np.einsum("nkt,nlt->nkl", Z[:, :, s:s+WIN], Z[:, :, s:s+WIN]) / WIN
                  for s in range(0, T - WIN + 1)], 1)
    d = C.shape[-1]
    C = C + 1e-6 * np.trace(C, axis1=-2, axis2=-1)[..., None, None] / d * np.eye(d)
    flat = C.reshape(-1, d, d)
    V = tsf(flat, mean_riemann(flat[::37])).reshape(C.shape[0], C.shape[1], -1)
    p1 = PCA(1).fit(V.reshape(-1, V.shape[2]))
    return np.stack([p1.transform(v)[:, 0] for v in V], 0)


def obs_geom_full(X, k=8):
    pc = PCA(NPCA_COV).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
    Z = np.stack([pc.transform(z.T).T for z in X], 0)
    T = Z.shape[2]
    C = np.stack([np.einsum("nkt,nlt->nkl", Z[:, :, s:s+WIN], Z[:, :, s:s+WIN]) / WIN
                  for s in range(0, T - WIN + 1)], 1)
    d = C.shape[-1]
    C = C + 1e-6 * np.trace(C, axis1=-2, axis2=-1)[..., None, None] / d * np.eye(d)
    flat = C.reshape(-1, d, d)
    V = tsf(flat, mean_riemann(flat[::37])).reshape(C.shape[0], C.shape[1], -1)
    pk = PCA(k).fit(V.reshape(-1, V.shape[2]))
    return np.stack([pk.transform(v) for v in V], 0).reshape(len(X), -1)


def winriem(Xtr, Xte, K, npca=20):
    pc = PCA(min(npca, len(Xtr)-2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    pr = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = pr(Xtr), pr(Xte)
    ed = np.linspace(0, A.shape[2], K+1).astype(int)
    fa, fb = [], []
    for i in range(K):
        Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
        Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
    return np.hstack(fa), np.hstack(fb)


def ev(F, y, seeds=3):
    acc = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(F, y):
            m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
            m.fit(F[tr], y[tr]); acc.append(balanced_accuracy_score(y[te], m.predict(F[te])))
    return float(np.mean(acc))


FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
NAMES = ["p_pc1_scalar", "p_geom_scalar", "p_geom_8ch", "winriem_full"]
R = {t: {k: [] for k in NAMES} for t in ["place", "ph11"]}
print("CEILING TEST: what does the OBSERVABLE itself carry?")
print("  the Type I network sees only e, driven by p -> its features cannot exceed p\n")
for f in FILES:
    Xall, ph = load(f)
    tasks = {"place": (np.isin(ph, STOPS), np.where(np.isin(ph[np.isin(ph, STOPS)], ["d", "t"]), 1, 0), 2),
             "ph11": (np.isin(ph, PHON), np.array([PHON.index(v) for v in ph[np.isin(ph, PHON)]]), 6)}
    for tn, (m, y, K) in tasks.items():
        X = Xall[m]
        R[tn]["p_pc1_scalar"].append(ev(obs_pc1(X), y))
        R[tn]["p_geom_scalar"].append(ev(obs_geom(X), y))
        R[tn]["p_geom_8ch"].append(ev(obs_geom_full(X, 8), y))
        acc = []
        for s in range(3):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                a, b = winriem(X[tr], X[te], K)
                mm = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
                mm.fit(a, y[tr]); acc.append(balanced_accuracy_score(y[te], mm.predict(b)))
        R[tn]["winriem_full"].append(float(np.mean(acc)))
    print("  %-8s ph11 %s" % (os.path.basename(f).split("_")[0],
          " ".join("%.3f" % R["ph11"][k][-1] for k in NAMES)), flush=True)

for tn, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tn, ch))
    for k in NAMES:
        a = np.array(R[tn][k]); _, p = ttest_1samp(a, ch)
        print("  %-16s %.3f +- %.3f  (%d/%d)  p=%.6f %s" % (
            k, a.mean(), a.std(), int((a > ch).sum()), len(a), p, "**" if p < 0.05 and a.mean() > ch else ""))
