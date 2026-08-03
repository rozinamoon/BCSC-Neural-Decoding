import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
GRID = {"place": [(d, k) for d in [20, 30, 40] for k in [2, 3]],
        "ph11": [(d, k) for d in [15, 20, 25] for k in [4, 6]]}


def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    k = ((meta.task == "single-phoneme") & meta.tms_target.str.startswith("control")).values
    bm = (t >= -0.2) & (t <= -0.11); wm = (t >= 0.0) & (t <= 0.5)
    Xk = X[k]
    return (Xk - Xk[:, :, bm].mean(2, keepdims=True))[:, :, wm], meta.phoneme.values[k]


def winriem(Xtr, Xte, K, npca):
    pc = PCA(min(npca, len(Xtr) - 2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    pr = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = pr(Xtr), pr(Xte)
    ed = np.linspace(0, A.shape[2], K + 1).astype(int)
    fa, fb = [], []
    for i in range(K):
        Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
        Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
    return np.hstack(fa), np.hstack(fb)


def reservoir(X, seed=0, npca=20, n_res=200, sub=20):
    rng = np.random.default_rng(seed)
    pc = PCA(npca).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
    U = np.stack([pc.transform(z.T) for z in X], 0)
    U = (U - U.mean((0, 1))) / (U.std((0, 1)) + 1e-9)
    W = rng.normal(0, 1, (n_res, n_res)) * (rng.random((n_res, n_res)) < 0.1)
    W = W / np.max(np.abs(np.linalg.eigvals(W)))
    Win = rng.normal(0, 1, (n_res, npca)) * 0.5
    leaks = np.exp(rng.uniform(np.log(0.02), np.log(0.9), n_res))[None, :]
    h = np.zeros((len(U), n_res)); out = []
    for ti in range(U.shape[1]):
        h = (1 - leaks) * h + leaks * np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
        if ti % sub == 0: out.append(h.copy())
    return np.stack(out, 1).reshape(len(U), -1)


def ev(Ftr, ytr, Fte, yte):
    m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    m.fit(Ftr, ytr); return balanced_accuracy_score(yte, m.predict(Fte))


def cat(bt, be):
    sc = [StandardScaler().fit(x) for x in bt]
    return (np.hstack([s.transform(x) for s, x in zip(sc, bt)]),
            np.hstack([s.transform(x) for s, x in zip(sc, be)]))


FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
RES = {t: {g: [] for g in GRID[t]} for t in GRID}
RESR = {t: {g: [] for g in GRID[t]} for t in GRID}
print("dimension x window sweep (PCA basis, LDA-shrinkage), with and without reservoir\n")
for f in FILES:
    Xall, ph = load(f)
    tasks = {"place": (np.isin(ph, STOPS),
                       np.where(np.isin(ph[np.isin(ph, STOPS)], ["d", "t"]), 1, 0)),
             "ph11": (np.isin(ph, PHON),
                      np.array([PHON.index(v) for v in ph[np.isin(ph, PHON)]]))}
    for tn, (m, y) in tasks.items():
        X = Xall[m]
        R = reservoir(X)
        acc = {g: [] for g in GRID[tn]}
        accr = {g: [] for g in GRID[tn]}
        for sd in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
                for (d, k) in GRID[tn]:
                    a, b = winriem(X[tr], X[te], k, d)
                    acc[(d, k)].append(ev(a, y[tr], b, y[te]))
                    A, B = cat([a, R[tr]], [b, R[te]])
                    accr[(d, k)].append(ev(A, y[tr], B, y[te]))
        for g in GRID[tn]:
            RES[tn][g].append(float(np.mean(acc[g])))
            RESR[tn][g].append(float(np.mean(accr[g])))
    print("  %s done" % os.path.basename(f).split("_")[0], flush=True)

for tn, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tn, ch))
    rows = sorted(GRID[tn], key=lambda g: -np.mean(RESR[tn][g]))
    print("  %-12s %10s %10s" % ("npca,K", "riem", "riem+res"))
    for g in rows:
        a, b = np.array(RES[tn][g]), np.array(RESR[tn][g])
        print("  d=%2d K=%d      %10.3f %10.3f" % (g[0], g[1], a.mean(), b.mean()))
    best_r = max(GRID[tn], key=lambda g: np.mean(RES[tn][g]))
    best_rr = max(GRID[tn], key=lambda g: np.mean(RESR[tn][g]))
    print("  BEST riem      d=%d K=%d -> %.3f" % (*best_r, np.mean(RES[tn][best_r])))
    print("  BEST riem+res  d=%d K=%d -> %.3f" % (*best_rr, np.mean(RESR[tn][best_rr])))
    _, p = ttest_rel(RESR[tn][best_rr], RES[tn][best_r])
    print("  reservoir gain %+.3f (p=%.4f)" % (
        np.mean(RESR[tn][best_rr]) - np.mean(RES[tn][best_r]), p))
