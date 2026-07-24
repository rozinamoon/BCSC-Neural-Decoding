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
from pyriemann.spatialfilters import Xdawn

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
D_IN, N_RES, LEAK = 20, 200, 0.3

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df.phoneme.values[m]

def winriem_feats(Xtr, Xte, K, npca=20):
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

def smooth(X3, leak=LEAK):
    out = np.empty_like(X3); s = np.zeros(X3.shape[:2])
    for ti in range(X3.shape[2]):
        s = (1-leak)*s + leak*X3[:, :, ti]; out[:, :, ti] = s
    return out

def make_esn(seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.1)
    W = W / np.max(np.abs(np.linalg.eigvals(W)))
    return W, rng.normal(0, 1, (N_RES, D_IN)) * 0.5

def esn_feats(Xtr, Xte, W, Win):
    pc = PCA(D_IN).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    def run(Z):
        U = np.stack([pc.transform(z.T) for z in Z], 0)
        n, T, _ = U.shape; h = np.zeros((n, N_RES)); out = []
        for ti in range(T):
            h = (1-LEAK)*h + LEAK*np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
            if ti % 20 == 0: out.append(h.copy())
        return np.stack(out, 1).reshape(n, -1)
    return run(Xtr), run(Xte)

def base_probs(Xtr, ytr, Xte, K, classes, W, Win):
    P = {}
    def fit_prob(Ftr, Fte):
        c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5))
        c.fit(Ftr, ytr)
        out = np.full((len(Fte), len(classes)), 1e-6)
        pp = c.predict_proba(Fte)
        for j, cl in enumerate(c.classes_):
            out[:, list(classes).index(cl)] = pp[:, j]
        return out
    a, b = winriem_feats(Xtr, Xte, K); P["winriem"] = fit_prob(a, b)
    xd = Xdawn(nfilter=4, estimator="oas").fit(Xtr, ytr)
    Ztr, Zte = xd.transform(Xtr), xd.transform(Xte)
    P["xdawn"] = fit_prob(Ztr[:, :, ::10].reshape(len(Xtr), -1), Zte[:, :, ::10].reshape(len(Xte), -1))
    a, b = esn_feats(Xtr, Xte, W, Win); P["reservoir"] = fit_prob(a, b)
    St, Se = smooth(Xtr), smooth(Xte)
    P["smooth"] = fit_prob(St[:, :, ::10].reshape(len(Xtr), -1), Se[:, :, ::10].reshape(len(Xte), -1))
    P["voltage"] = fit_prob(Xtr[:, :, ::10].reshape(len(Xtr), -1), Xte[:, :, ::10].reshape(len(Xte), -1))
    return P

METHODS = ["winriem", "xdawn", "reservoir", "smooth", "voltage"]
RULES = ["best_single", "SUM", "PRODUCT", "PRODUCT_T2", "max_conf", "STACKING"]
W, Win = make_esn(0)
RES = {t: {r: [] for r in RULES} for t in ["place", "ph11"]}
print("COMBINATION RULES: sum vs PRODUCT (multiplying information) vs stacking\n")
for f in FILES:
    Xall, ph = load(f)
    k1 = np.isin(ph, ["b", "p", "d", "t"]); k2 = np.isin(ph, PHON)
    tasks = {"place": (Xall[k1], np.where(np.isin(ph[k1], ["d", "t"]), 1, 0), 2),
             "ph11": (Xall[k2], np.array([PHON.index(v) for v in ph[k2]]), 6)}
    for tname, (X, y, K) in tasks.items():
        classes = np.unique(y)
        acc = {r: [] for r in RULES}
        for s in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                P = base_probs(X[tr], y[tr], X[te], K, classes, W, Win)
                Ps = np.stack([P[m] for m in METHODS], 0)
                acc["best_single"].append(balanced_accuracy_score(y[te], classes[np.argmax(P["winriem"], 1)]))
                acc["SUM"].append(balanced_accuracy_score(y[te], classes[np.argmax(Ps.mean(0), 1)]))
                lp = np.log(np.clip(Ps, 1e-9, 1)).sum(0)
                acc["PRODUCT"].append(balanced_accuracy_score(y[te], classes[np.argmax(lp, 1)]))
                acc["PRODUCT_T2"].append(balanced_accuracy_score(y[te], classes[np.argmax(lp/2.0, 1)]))
                conf = Ps.max(2)
                pick = np.argmax(conf, 0)
                pred = np.array([classes[np.argmax(Ps[pick[i], i])] for i in range(len(y[te]))])
                acc["max_conf"].append(balanced_accuracy_score(y[te], pred))
                inner = StratifiedKFold(4, shuffle=True, random_state=0)
                oof = np.zeros((len(tr), len(METHODS)*len(classes)))
                for itr, ite in inner.split(X[tr], y[tr]):
                    Pi = base_probs(X[tr][itr], y[tr][itr], X[tr][ite], K, classes, W, Win)
                    oof[ite] = np.hstack([Pi[m] for m in METHODS])
                meta = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
                meta.fit(oof, y[tr])
                acc["STACKING"].append(balanced_accuracy_score(y[te], meta.predict(np.hstack([P[m] for m in METHODS]))))
        for r in RULES: RES[tname][r].append(float(np.mean(acc[r])))
    print("  %s done" % os.path.basename(f).split("_")[0], flush=True)

for tname, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tname, ch))
    base = np.array(RES[tname]["best_single"])
    for r in RULES:
        a = np.array(RES[tname][r]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-12s %.3f  vs-chance p=%.6f  vs-best_single %+.3f (p=%.4f) %s" % (
            r, a.mean(), p0, a.mean()-base.mean(), pb, "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
