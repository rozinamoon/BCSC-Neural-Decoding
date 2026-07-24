import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.linear_model import LogisticRegression
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
WMIN, WMAX = 0.0, 0.5
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
N_RES, LEAK = 200, 0.3

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df.phoneme.values[m]

def smooth(X3, leak=LEAK):
    out = np.empty_like(X3); s = np.zeros(X3.shape[:2])
    for ti in range(X3.shape[2]):
        s = (1-leak)*s + leak*X3[:, :, ti]; out[:, :, ti] = s
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

def make_esn(d_in, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.1)
    W = W / np.max(np.abs(np.linalg.eigvals(W)))
    return W, rng.normal(0, 1, (N_RES, d_in)) * 0.5

def run_esn(U, W, Win):
    n, T, _ = U.shape; h = np.zeros((n, N_RES)); out = []
    for ti in range(T):
        h = (1-LEAK)*h + LEAK*np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
        if ti % 20 == 0: out.append(h.copy())
    return np.stack(out, 1).reshape(n, -1)

def cat(blocks_tr, blocks_te):
    scs = [StandardScaler().fit(b) for b in blocks_tr]
    return (np.hstack([s.transform(b) for s, b in zip(scs, blocks_tr)]),
            np.hstack([s.transform(b) for s, b in zip(scs, blocks_te)]))

def ev(Ftr, ytr, Fte, yte, kind="lr"):
    c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5)) \
        if kind == "lr" else make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    c.fit(Ftr, ytr)
    return balanced_accuracy_score(yte, c.predict(Fte))

NAMES = ["winriem_ONLY", "riem+xdawn", "riem+res", "riem+xdawn+res", "riem+xdawn+res+smooth", "ALL_lda"]
RES = {t: {k: [] for k in NAMES} for t in ["place", "ph11"]}
print("DIRECT COMPARISON: concatenation fusion vs previous best (same CV splits)\n")
for f in FILES:
    Xall, ph = load(f)
    k1 = np.isin(ph, ["b", "p", "d", "t"]); k2 = np.isin(ph, PHON)
    tasks = {"place": (Xall[k1], np.where(np.isin(ph[k1], ["d", "t"]), 1, 0), 2),
             "ph11": (Xall[k2], np.array([PHON.index(v) for v in ph[k2]]), 6)}
    for tname, (X, y, K) in tasks.items():
        acc = {k: [] for k in NAMES}
        for s in range(3):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                Rtr, Rte = winriem(X[tr], X[te], K)
                xd = Xdawn(nfilter=4, estimator="oas").fit(X[tr], y[tr])
                Ztr = xd.transform(X[tr])[:, :, ::10].reshape(len(tr), -1)
                Zte = xd.transform(X[te])[:, :, ::10].reshape(len(te), -1)
                pc = PCA(20).fit(X[tr].transpose(0, 2, 1).reshape(-1, X.shape[1]))
                Utr = np.stack([pc.transform(z.T) for z in X[tr]], 0)
                Ute = np.stack([pc.transform(z.T) for z in X[te]], 0)
                W1, Win1 = make_esn(20, 0)
                Str, Ste = run_esn(Utr, W1, Win1), run_esn(Ute, W1, Win1)
                Smt = smooth(X[tr])[:, :, ::10].reshape(len(tr), -1)
                Sme = smooth(X[te])[:, :, ::10].reshape(len(te), -1)
                acc["winriem_ONLY"].append(ev(Rtr, y[tr], Rte, y[te]))
                a, b = cat([Rtr, Ztr], [Rte, Zte]); acc["riem+xdawn"].append(ev(a, y[tr], b, y[te]))
                a, b = cat([Rtr, Str], [Rte, Ste]); acc["riem+res"].append(ev(a, y[tr], b, y[te]))
                a, b = cat([Rtr, Ztr, Str], [Rte, Zte, Ste]); acc["riem+xdawn+res"].append(ev(a, y[tr], b, y[te]))
                a2, b2 = cat([Rtr, Ztr, Str, Smt], [Rte, Zte, Ste, Sme])
                acc["riem+xdawn+res+smooth"].append(ev(a2, y[tr], b2, y[te]))
                acc["ALL_lda"].append(ev(a2, y[tr], b2, y[te], kind="lda"))
        for k in NAMES: RES[tname][k].append(float(np.mean(acc[k])))
    print("  %-8s place %.3f->%.3f | ph11 %.3f->%.3f" % (
        os.path.basename(f).split("_")[0], RES["place"]["winriem_ONLY"][-1], RES["place"]["riem+xdawn+res"][-1],
        RES["ph11"]["winriem_ONLY"][-1], RES["ph11"]["riem+xdawn+res"][-1]), flush=True)

for tname, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tname, ch))
    base = np.array(RES[tname]["winriem_ONLY"])
    for k in NAMES:
        a = np.array(RES[tname][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-24s %.3f  vs-chance p=%.7f  vs-winriem %+.3f (p=%.4f) %s" % (
            k, a.mean(), p0, a.mean()-base.mean(), pb, "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
