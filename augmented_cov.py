import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
CONFIGS = [(1, 0, 20), (2, 8, 12), (3, 8, 10), (4, 8, 8), (3, 16, 10), (4, 4, 8), (5, 6, 6)]

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df.phoneme.values[m]

def augmented_cov(Xtr, Xte, p_lag, tau, npca):
    pc = PCA(npca).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    proj = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = proj(Xtr), proj(Xte)
    def embed(S):
        if p_lag == 1:
            return S
        T = S.shape[2]; span = (p_lag - 1) * tau
        return np.concatenate([S[:, :, i*tau:T - span + i*tau] for i in range(p_lag)], axis=1)
    Ca = Covariances("oas").fit_transform(embed(A))
    Cb = Covariances("oas").fit_transform(embed(B))
    ts = TangentSpace(metric="riemann").fit(Ca)
    return ts.transform(Ca), ts.transform(Cb)

def win_riem(Xtr, Xte, K=4, npca=20):
    pc = PCA(npca).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    proj = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = proj(Xtr), proj(Xte)
    T = A.shape[2]; ed = np.linspace(0, T, K+1).astype(int)
    fa, fb = [], []
    for i in range(K):
        Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
        Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
    return np.hstack(fa), np.hstack(fb)

def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5))

NAMES = ["p%d_tau%d" % (p, t) for p, t, _ in CONFIGS] + ["winriem_K4"]
RES = {t: {k: [] for k in NAMES} for t in ["place", "ph11"]}
print("AUGMENTED COVARIANCE: delay-embedded state -> Riemannian geometry\n")
print("  (covariance of delay-embedded signal encodes cross-lag correlations = temporal structure)\n")
for f in FILES:
    Xall, ph = load(f)
    k1 = np.isin(ph, ["b", "p", "d", "t"]); k2 = np.isin(ph, PHON)
    tasks = {"place": (Xall[k1], np.where(np.isin(ph[k1], ["d", "t"]), 1, 0)),
             "ph11": (Xall[k2], np.array([PHON.index(v) for v in ph[k2]]))}
    for tname, (X, y) in tasks.items():
        acc = {k: [] for k in NAMES}
        for s in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                for (p_lag, tau, npca), nm in zip(CONFIGS, NAMES):
                    a, b = augmented_cov(X[tr], X[te], p_lag, tau, npca)
                    c = lr(); c.fit(a, y[tr]); acc[nm].append(balanced_accuracy_score(y[te], c.predict(b)))
                a, b = win_riem(X[tr], X[te])
                c = lr(); c.fit(a, y[tr]); acc["winriem_K4"].append(balanced_accuracy_score(y[te], c.predict(b)))
        for k in NAMES: RES[tname][k].append(float(np.mean(acc[k])))
    print("  %-8s place %s" % (os.path.basename(f).split("_")[0],
          " ".join("%.3f" % RES["place"][k][-1] for k in NAMES)), flush=True)

for tname, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tname, ch))
    base = np.array(RES[tname]["winriem_K4"])
    for k in NAMES:
        a = np.array(RES[tname][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-12s %.3f  vs-chance p=%.6f  vs-winriem %+.3f (p=%.4f)" % (k, a.mean(), p0, a.mean()-base.mean(), pb))
