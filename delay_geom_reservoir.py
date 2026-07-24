import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.spatialfilters import Xdawn

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
N_RES, D_IN, SUB = 200, 20, 20
DELAYS = np.array([1, 3, 7, 15, 31])

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    k = ((meta.task == "single-phoneme") & meta.tms_target.str.startswith("control")).values
    bm = (t >= -0.2) & (t <= -0.11); wm = (t >= 0.0) & (t <= 0.5)
    Xk = X[k]
    return (Xk - Xk[:, :, bm].mean(2, keepdims=True))[:, :, wm], meta.phoneme.values[k]

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

def rand_W(seed=0, rho=1.0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.1)
    return W / np.max(np.abs(np.linalg.eigvals(W))) * rho

def geom_W(Xtr, ytr, seed=0, rho=1.0):
    pc = PCA(D_IN).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    P = np.stack([pc.transform(z.T).T for z in Xtr], 0)
    cls = np.unique(ytr)
    Cs = [Covariances("oas").fit_transform(P[ytr == c]).mean(0) for c in cls]
    Cb = np.mean(Cs, 0)
    Cw = np.mean([Covariances("oas").fit_transform(P[ytr == c]).mean(0) for c in cls], 0)
    ev, V = eigh(Cb, Cw + 1e-6 * np.eye(Cw.shape[0]))
    V = V[:, np.argsort(ev)[::-1]]
    rng = np.random.default_rng(seed)
    M = rng.normal(0, 1, (N_RES, D_IN)) @ V.T
    W = M @ M.T
    W = W * (rng.random((N_RES, N_RES)) < 0.3)
    W = 0.5 * (W + W.T)
    return W / (np.max(np.abs(np.linalg.eigvals(W))) + 1e-12) * rho

def run_tanh(U, W, Win, leaks):
    n, T, _ = U.shape
    h = np.zeros((n, N_RES)); out = []
    for ti in range(T):
        h = (1-leaks)*h + leaks*np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
        if ti % SUB == 0: out.append(h.copy())
    return np.stack(out, 1).reshape(n, -1)

def run_delay(U, W, Win, tau_idx, leak=0.3):
    n, T, _ = U.shape
    buf = np.zeros((n, DELAYS.max()+1, N_RES))
    out = []
    for ti in range(T):
        hd = buf[:, -tau_idx, np.arange(N_RES)]
        pre = np.tanh(hd @ W.T + U[:, ti, :] @ Win.T)
        h = (1-leak)*buf[:, -1, :] + leak*pre
        buf = np.concatenate([buf[:, 1:, :], h[:, None, :]], 1)
        if ti % SUB == 0: out.append(h.copy())
    return np.stack(out, 1).reshape(n, -1)

def prep(X):
    pc = PCA(D_IN).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
    U = np.stack([pc.transform(z.T) for z in X], 0)
    return (U - U.mean((0, 1))) / (U.std((0, 1)) + 1e-9)

def ev(Ftr, ytr, Fte, yte):
    c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    c.fit(Ftr, ytr); return balanced_accuracy_score(yte, c.predict(Fte))

def cat(bt, be):
    sc = [StandardScaler().fit(b) for b in bt]
    return (np.hstack([s.transform(b) for s, b in zip(sc, bt)]),
            np.hstack([s.transform(b) for s, b in zip(sc, be)]))

rng0 = np.random.default_rng(0)
LEAKS = np.exp(rng0.uniform(np.log(0.02), np.log(0.9), N_RES))[None, :]
TAU_IDX = rng0.choice(DELAYS, N_RES)
WIN_R = rng0.normal(0, 1, (N_RES, D_IN)) * 0.5
W_RAND = rand_W(0)

NAMES = ["winriem", "res_multitau", "RES_DELAY", "RES_GEOMW", "RES_DELAY_GEOMW",
         "win+multitau", "win+DELAY", "win+GEOMW", "win+DELAY+GEOMW"]
RES = {t: {k: [] for k in NAMES} for t in ["place", "ph11"]}
print("Delay neurons and geometry-derived W inside the reservoir\n")
for f in FILES:
    Xall, ph = load(f)
    tasks = {"place": (Xall[np.isin(ph, STOPS)],
                       np.where(np.isin(ph[np.isin(ph, STOPS)], ["d", "t"]), 1, 0), 2),
             "ph11": (Xall[np.isin(ph, PHON)],
                      np.array([PHON.index(v) for v in ph[np.isin(ph, PHON)]]), 6)}
    for tn, (X, y, K) in tasks.items():
        U = prep(X)
        F_mt = run_tanh(U, W_RAND, WIN_R, LEAKS)
        F_dl = run_delay(U, W_RAND, WIN_R, TAU_IDX)
        acc = {k: [] for k in NAMES}
        for s in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                Rtr, Rte = winriem(X[tr], X[te], K)
                acc["winriem"].append(ev(Rtr, y[tr], Rte, y[te]))
                Wg = geom_W(X[tr], y[tr], seed=s)
                F_gw = run_tanh(U, Wg, WIN_R, LEAKS)
                F_dg = run_delay(U, Wg, WIN_R, TAU_IDX)
                for nm, F in [("res_multitau", F_mt), ("RES_DELAY", F_dl),
                              ("RES_GEOMW", F_gw), ("RES_DELAY_GEOMW", F_dg)]:
                    acc[nm].append(ev(F[tr], y[tr], F[te], y[te]))
                for nm, F in [("win+multitau", F_mt), ("win+DELAY", F_dl),
                              ("win+GEOMW", F_gw), ("win+DELAY+GEOMW", F_dg)]:
                    a, b = cat([Rtr, F[tr]], [Rte, F[te]])
                    acc[nm].append(ev(a, y[tr], b, y[te]))
        for k in NAMES: RES[tn][k].append(float(np.mean(acc[k])))
    print("  %-8s ph11 %s" % (os.path.basename(f).split("_")[0],
          " ".join("%.3f" % RES["ph11"][k][-1] for k in NAMES)), flush=True)

for tn, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tn, ch))
    base = np.array(RES[tn]["winriem"])
    for k in NAMES:
        a = np.array(RES[tn][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-18s %.3f  vs-chance p=%.6f  vs-winriem %+.3f (p=%.4f) %s" % (
            k, a.mean(), p0, a.mean()-base.mean(), pb,
            "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
