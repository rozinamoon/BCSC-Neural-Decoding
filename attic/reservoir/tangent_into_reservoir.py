import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
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
WMIN, WMAX = 0.0, 0.5
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
D_TAN, WLEN, STRIDE = 8, 50, 12
N_RES, LEAK, SUB = 200, 0.3, 2

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & df.tms_target.str.startswith("control")).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df.phoneme.values[m]

def sliding_covs(Xp):
    n, d, T = Xp.shape
    starts = list(range(0, T - WLEN + 1, STRIDE))
    C = np.empty((n, len(starts), d, d))
    for i, s0 in enumerate(starts):
        C[:, i] = Covariances("oas").fit_transform(Xp[:, :, s0:s0+WLEN])
    return C

def tangent_series(Ctr, Cte):
    n_tr, W, d, _ = Ctr.shape
    ts = TangentSpace(metric="logeuclid").fit(Ctr.reshape(-1, d, d))
    A = ts.transform(Ctr.reshape(-1, d, d)).reshape(n_tr, W, -1)
    B = ts.transform(Cte.reshape(-1, d, d)).reshape(len(Cte), W, -1)
    return A, B

def make_esn(d_in, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.1)
    W = W / np.max(np.abs(np.linalg.eigvals(W)))
    return W, rng.normal(0, 1, (N_RES, d_in)) * 0.5

def run_esn(U, W, Win, sub=None):
    n, T, _ = U.shape
    h = np.zeros((n, N_RES)); out = []
    for ti in range(T):
        h = (1-LEAK)*h + LEAK*np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
        if ti % (sub or SUB) == 0: out.append(h.copy())
    return np.stack(out, 1).reshape(n, -1)

def winriem(Xtr, Xte, K, npca=20):
    pc = PCA(npca).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    proj = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = proj(Xtr), proj(Xte)
    T = A.shape[2]; ed = np.linspace(0, T, K+1).astype(int)
    fa, fb = [], []
    for i in range(K):
        Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
        Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
        t_ = TangentSpace(metric="riemann").fit(Ca)
        fa.append(t_.transform(Ca)); fb.append(t_.transform(Cb))
    return np.hstack(fa), np.hstack(fb)

def ev(Ftr, ytr, Fte, yte):
    c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    c.fit(Ftr, ytr); return balanced_accuracy_score(yte, c.predict(Fte))

NAMES = ["winriem", "TANGENT_SERIES", "TANGENT_into_RESERVOIR", "raw_into_reservoir", "TANsr+winriem"]
RES = {t: {k: [] for k in NAMES} for t in ["place", "ph11"]}
print("(c) Riemannian tangent series AS RESERVOIR INPUT\n")
print("  EEG -> sliding-window cov (L=%d,s=%d) -> tangent -> time series -> reservoir\n" % (WLEN, STRIDE))
for f in FILES:
    Xall, ph = load(f)
    k1 = np.isin(ph, ["b", "p", "d", "t"]); k2 = np.isin(ph, PHON)
    tasks = {"place": (Xall[k1], np.where(np.isin(ph[k1], ["d", "t"]), 1, 0), 2),
             "ph11": (Xall[k2], np.array([PHON.index(v) for v in ph[k2]]), 6)}
    for tname, (X, y, K) in tasks.items():
        pcg = PCA(D_TAN).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
        Xp = np.stack([pcg.transform(z.T).T for z in X], 0)
        COV = sliding_covs(Xp)
        pcr = PCA(20).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
        Ur = np.stack([pcr.transform(z.T) for z in X], 0)
        Ur = (Ur - Ur.mean((0, 1))) / (Ur.std((0, 1)) + 1e-9)
        acc = {k: [] for k in NAMES}
        for s in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                Rtr, Rte = winriem(X[tr], X[te], K)
                acc["winriem"].append(ev(Rtr, y[tr], Rte, y[te]))
                Atr, Ate = tangent_series(COV[tr], COV[te])
                acc["TANGENT_SERIES"].append(ev(Atr.reshape(len(tr), -1), y[tr], Ate.reshape(len(te), -1), y[te]))
                mu, sd = Atr.mean((0, 1)), Atr.std((0, 1)) + 1e-9
                W1, Win1 = make_esn(Atr.shape[2], 0)
                Ftr = run_esn((Atr - mu)/sd, W1, Win1); Fte = run_esn((Ate - mu)/sd, W1, Win1)
                acc["TANGENT_into_RESERVOIR"].append(ev(Ftr, y[tr], Fte, y[te]))
                W2, Win2 = make_esn(Ur.shape[2], 0)
                acc["raw_into_reservoir"].append(ev(run_esn(Ur[tr], W2, Win2, 20), y[tr], run_esn(Ur[te], W2, Win2, 20), y[te]))
                s1 = StandardScaler().fit(Rtr); s2 = StandardScaler().fit(Atr.reshape(len(tr), -1))
                a = np.hstack([s1.transform(Rtr), s2.transform(Atr.reshape(len(tr), -1))])
                b = np.hstack([s1.transform(Rte), s2.transform(Ate.reshape(len(te), -1))])
                acc["TANsr+winriem"].append(ev(a, y[tr], b, y[te]))
        for k in NAMES: RES[tname][k].append(float(np.mean(acc[k])))
    print("  %-8s place %s | ph11 %s" % (os.path.basename(f).split("_")[0],
          " ".join("%.3f" % RES["place"][k][-1] for k in NAMES),
          " ".join("%.3f" % RES["ph11"][k][-1] for k in NAMES)), flush=True)

for tname, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tname, ch))
    base = np.array(RES[tname]["winriem"])
    for k in NAMES:
        a = np.array(RES[tname][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-24s %.3f  vs-chance p=%.7f  vs-winriem %+.3f (p=%.4f) %s" % (
            k, a.mean(), p0, a.mean()-base.mean(), pb, "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
