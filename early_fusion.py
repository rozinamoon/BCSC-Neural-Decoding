import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
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

def kern(Ftr, Fte):
    sc = StandardScaler().fit(Ftr)
    A, B = sc.transform(Ftr), sc.transform(Fte)
    g = 1.0 / (A.shape[1] * A.var() + 1e-12)
    return rbf_kernel(A, A, gamma=g), rbf_kernel(B, A, gamma=g)

def svc_eval(Ktr, Kte, ytr, yte):
    c = SVC(kernel="precomputed", class_weight="balanced", C=1.0)
    c.fit(Ktr, ytr)
    return balanced_accuracy_score(yte, c.predict(Kte))

NAMES = ["riem_only", "SUM_KERNEL", "PRODUCT_KERNEL", "CONCAT_linear", "res_single", "RES_MULTIINPUT", "RES_MULTI+RIEM"]
RES = {t: {k: [] for k in NAMES} for t in ["place", "ph11"]}
print("EARLY FUSION: product KERNEL + multi-representation RESERVOIR\n")
for f in FILES:
    Xall, ph = load(f)
    k1 = np.isin(ph, ["b", "p", "d", "t"]); k2 = np.isin(ph, PHON)
    tasks = {"place": (Xall[k1], np.where(np.isin(ph[k1], ["d", "t"]), 1, 0), 2),
             "ph11": (Xall[k2], np.array([PHON.index(v) for v in ph[k2]]), 6)}
    for tname, (X, y, K) in tasks.items():
        acc = {k: [] for k in NAMES}
        for s in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                Rtr, Rte = winriem(X[tr], X[te], K)
                xd = Xdawn(nfilter=4, estimator="oas").fit(X[tr], y[tr])
                Ztr, Zte = xd.transform(X[tr]), xd.transform(X[te])
                Xtr_f = Ztr[:, :, ::10].reshape(len(tr), -1); Xte_f = Zte[:, :, ::10].reshape(len(te), -1)
                pc = PCA(20).fit(X[tr].transpose(0, 2, 1).reshape(-1, X.shape[1]))
                Utr = np.stack([pc.transform(z.T) for z in X[tr]], 0)
                Ute = np.stack([pc.transform(z.T) for z in X[te]], 0)
                W1, Win1 = make_esn(20, 0)
                Str, Ste = run_esn(Utr, W1, Win1), run_esn(Ute, W1, Win1)
                K1a, K1b = kern(Rtr, Rte); K2a, K2b = kern(Xtr_f, Xte_f); K3a, K3b = kern(Str, Ste)
                acc["riem_only"].append(svc_eval(K1a, K1b, y[tr], y[te]))
                acc["SUM_KERNEL"].append(svc_eval((K1a+K2a+K3a)/3, (K1b+K2b+K3b)/3, y[tr], y[te]))
                acc["PRODUCT_KERNEL"].append(svc_eval(K1a*K2a*K3a, K1b*K2b*K3b, y[tr], y[te]))
                cat_tr = np.hstack([StandardScaler().fit_transform(v) for v in [Rtr, Xtr_f, Str]])
                sc = [StandardScaler().fit(v) for v in [Rtr, Xtr_f, Str]]
                cat_te = np.hstack([s_.transform(v) for s_, v in zip(sc, [Rte, Xte_f, Ste])])
                c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5))
                c.fit(cat_tr, y[tr]); acc["CONCAT_linear"].append(balanced_accuracy_score(y[te], c.predict(cat_te)))
                c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5))
                c.fit(Str, y[tr]); acc["res_single"].append(balanced_accuracy_score(y[te], c.predict(Ste)))
                Smt, Sme = smooth(X[tr]), smooth(X[te])
                pcs = PCA(10).fit(Smt.transpose(0, 2, 1).reshape(-1, X.shape[1]))
                mtr = np.concatenate([Utr, np.stack([pcs.transform(z.T) for z in Smt], 0),
                                      np.transpose(Ztr, (0, 2, 1))], axis=2)
                mte = np.concatenate([Ute, np.stack([pcs.transform(z.T) for z in Sme], 0),
                                      np.transpose(Zte, (0, 2, 1))], axis=2)
                W2, Win2 = make_esn(mtr.shape[2], 0)
                Mtr, Mte = run_esn(mtr, W2, Win2), run_esn(mte, W2, Win2)
                c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5))
                c.fit(Mtr, y[tr]); acc["RES_MULTIINPUT"].append(balanced_accuracy_score(y[te], c.predict(Mte)))
                ro_tr = np.hstack([StandardScaler().fit_transform(Mtr), StandardScaler().fit_transform(Rtr)])
                s1 = StandardScaler().fit(Mtr); s2 = StandardScaler().fit(Rtr)
                ro_te = np.hstack([s1.transform(Mte), s2.transform(Rte)])
                c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5))
                c.fit(ro_tr, y[tr]); acc["RES_MULTI+RIEM"].append(balanced_accuracy_score(y[te], c.predict(ro_te)))
        for k in NAMES: RES[tname][k].append(float(np.mean(acc[k])))
    print("  %s done" % os.path.basename(f).split("_")[0], flush=True)

for tname, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tname, ch))
    base = np.array(RES[tname]["riem_only"])
    for k in NAMES:
        a = np.array(RES[tname][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-16s %.3f  vs-chance p=%.6f  vs-riem_only %+.3f (p=%.4f) %s" % (
            k, a.mean(), p0, a.mean()-base.mean(), pb, "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
