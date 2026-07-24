import warnings
warnings.filterwarnings("ignore")
import glob
import os
import collections
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import RidgeClassifierCV
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

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df.phoneme.values[m]

print("=== 1. CLASS BALANCE CHECK (single-phoneme control) ===")
counts_all = []
for f in FILES:
    _, ph = load(f)
    c = collections.Counter(ph[np.isin(ph, PHON)])
    v = np.array([c[p] for p in PHON])
    counts_all.append(v)
    print("  %-8s n=%3d  per-class %s  min=%d max=%d" % (
        os.path.basename(f).split("_")[0], v.sum(), list(v), v.min(), v.max()))
CA = np.array(counts_all)
print("  GROUP: total %d, per-class mean %.1f, imbalance ratio max/min = %.2f" % (
    CA.sum(), CA.mean(), CA.sum(0).max() / CA.sum(0).min()))

def rocket(X, n_kernels, seed=0):
    rng = np.random.default_rng(seed)
    n, C, T = X.shape
    out = np.empty((n, n_kernels * 2), np.float32)
    for k in range(n_kernels):
        L = rng.choice([7, 9, 11])
        ch = rng.choice(C, size=min(C, rng.choice([1, 2, 4])), replace=False)
        w = rng.normal(0, 1, (len(ch), L)); w = w - w.mean()
        max_d = max(1, int(np.log2((T - 1) / (L - 1))))
        dil = int(2 ** rng.uniform(0, max_d))
        span = (L - 1) * dil
        if span >= T: dil, span = 1, L - 1
        idx = np.arange(L) * dil
        seg = np.stack([X[:, ch, i:T - span + i] for i in idx], -1)
        conv = np.einsum("ncti,ci->nt", seg, w)
        b = rng.uniform(-1, 1) * (np.abs(conv).mean() + 1e-9)
        z = conv - b
        out[:, 2*k] = (z > 0).mean(1); out[:, 2*k+1] = z.max(1)
    return out

def winriem(Xtr, Xte, K, npca=20):
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

def ev(Ftr, ytr, Fte, yte):
    c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    c.fit(Ftr, ytr); return balanced_accuracy_score(yte, c.predict(Fte))

KERNELS = [500, 2000, 5000]
NAMES = ["winriem"] + ["RK%d" % k for k in KERNELS] + ["win+RK%d" % k for k in KERNELS]
RES = {k: [] for k in NAMES}
PERM = {"real": [], "null": []}
print("\n=== 2+3. ROCKET kernel sweep + PERMUTATION (11-phoneme) ===")
rng_g = np.random.default_rng(0)
for f in FILES:
    Xall, ph = load(f)
    k2 = np.isin(ph, PHON)
    X = Xall[k2]; y = np.array([PHON.index(v) for v in ph[k2]])
    pcg = PCA(12).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
    Xp = np.stack([pcg.transform(z.T).T for z in X], 0)
    RK = {k: rocket(Xp, k) for k in KERNELS}
    acc = {k: [] for k in NAMES}
    perm_r, perm_n = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        Rtr, Rte = winriem(X[tr], X[te], 6)
        acc["winriem"].append(ev(Rtr, y[tr], Rte, y[te]))
        s1 = StandardScaler().fit(Rtr)
        for k in KERNELS:
            acc["RK%d" % k].append(ev(RK[k][tr], y[tr], RK[k][te], y[te]))
            s2 = StandardScaler().fit(RK[k][tr])
            a = np.hstack([s1.transform(Rtr), s2.transform(RK[k][tr])])
            b = np.hstack([s1.transform(Rte), s2.transform(RK[k][te])])
            acc["win+RK%d" % k].append(ev(a, y[tr], b, y[te]))
        s2 = StandardScaler().fit(RK[500][tr])
        a = np.hstack([s1.transform(Rtr), s2.transform(RK[500][tr])])
        b = np.hstack([s1.transform(Rte), s2.transform(RK[500][te])])
        perm_r.append(ev(a, y[tr], b, y[te]))
        for _ in range(6):
            yp = rng_g.permutation(y[tr])
            perm_n.append(ev(a, yp, b, y[te]))
    for k in NAMES: RES[k].append(float(np.mean(acc[k])))
    PERM["real"].append(np.mean(perm_r)); PERM["null"].append(np.mean(perm_n))
    print("  %-8s win %.3f | RK500 %.3f RK2000 %.3f RK5000 %.3f | win+RK500 %.3f win+RK2000 %.3f win+RK5000 %.3f | null %.3f" % (
        os.path.basename(f).split("_")[0], RES["winriem"][-1], RES["RK500"][-1], RES["RK2000"][-1], RES["RK5000"][-1],
        RES["win+RK500"][-1], RES["win+RK2000"][-1], RES["win+RK5000"][-1], PERM["null"][-1]), flush=True)

print("\n=== GROUP (11-phoneme) ===")
base = np.array(RES["winriem"])
nullm = np.mean(PERM["null"])
print("  EMPIRICAL CHANCE (permuted labels): %.4f   [theoretical 1/11 = %.4f]" % (nullm, 1/11))
for k in NAMES:
    a = np.array(RES[k]); _, p0 = ttest_1samp(a, nullm); _, pb = ttest_rel(a, base)
    print("  %-14s %.3f  vs-empirical-null p=%.6f  vs-winriem %+.3f (p=%.4f) %s" % (
        k, a.mean(), p0, a.mean()-base.mean(), pb, "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
