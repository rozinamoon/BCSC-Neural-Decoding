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
D_IN, N_RES, LEAK, RHO = 20, 40, 0.3, 1.0

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w]
    ph = df.phoneme.values[m]
    k = np.isin(ph, ["b", "p", "d", "t"])
    return Xb[k], np.where(np.isin(ph[k], ["d", "t"]), 1, 0)

def make_esn(seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.2)
    W = W / np.max(np.abs(np.linalg.eigvals(W))) * RHO
    return W, rng.normal(0, 1, (N_RES, D_IN)) * 0.5

def esn_states(X, W, Win):
    p = PCA(D_IN).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
    U = np.stack([p.transform(x.T) for x in X], 0)
    U = (U - U.mean((0, 1))) / (U.std((0, 1)) + 1e-9)
    n, T, _ = U.shape; h = np.zeros((n, N_RES)); out = []
    for ti in range(T):
        h = (1-LEAK)*h + LEAK*np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
        out.append(h.copy())
    return np.stack(out, 2)

def tan(Ctr, Cte):
    ts = TangentSpace(metric="riemann").fit(Ctr)
    return ts.transform(Ctr), ts.transform(Cte)

def whole_cov(Str, Ste):
    return tan(Covariances("oas").fit_transform(Str), Covariances("oas").fit_transform(Ste))

def windowed_cov(Str, Ste, K):
    T = Str.shape[2]; edges = np.linspace(0, T, K+1).astype(int)
    ftr, fte = [], []
    for i in range(K):
        s, e = edges[i], edges[i+1]
        a, b = tan(Covariances("oas").fit_transform(Str[:, :, s:e]),
                   Covariances("oas").fit_transform(Ste[:, :, s:e]))
        ftr.append(a); fte.append(b)
    return np.hstack(ftr), np.hstack(fte)

def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))

W, Win = make_esn(0)
NAMES = ["raw_riem", "res_flat", "res_riem_whole", "res_riem_win4", "res_riem_win8", "raw_riem_win4"]
R = {k: [] for k in NAMES}
print("Riemannian applied DIRECTLY in reservoir (chaos) space | N_res=%d, rho=%.1f\n" % (N_RES, RHO))
for f in FILES:
    X, y = load(f)
    S = esn_states(X, W, Win)
    acc = {k: [] for k in NAMES}
    for s in range(3):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            p = PCA(D_IN).fit(X[tr].transpose(0, 2, 1).reshape(-1, X.shape[1]))
            pr = lambda Z: np.stack([p.transform(z.T).T for z in Z], 0)
            a, b = whole_cov(pr(X[tr]), pr(X[te]))
            c = lr(); c.fit(a, y[tr]); acc["raw_riem"].append(balanced_accuracy_score(y[te], c.predict(b)))
            a, b = windowed_cov(pr(X[tr]), pr(X[te]), 4)
            c = lr(); c.fit(a, y[tr]); acc["raw_riem_win4"].append(balanced_accuracy_score(y[te], c.predict(b)))
            ftr = S[tr][:, :, ::20].reshape(len(tr), -1); fte = S[te][:, :, ::20].reshape(len(te), -1)
            c = lr(); c.fit(ftr, y[tr]); acc["res_flat"].append(balanced_accuracy_score(y[te], c.predict(fte)))
            a, b = whole_cov(S[tr], S[te])
            c = lr(); c.fit(a, y[tr]); acc["res_riem_whole"].append(balanced_accuracy_score(y[te], c.predict(b)))
            for K in [4, 8]:
                a, b = windowed_cov(S[tr], S[te], K)
                c = lr(); c.fit(a, y[tr]); acc["res_riem_win%d" % K].append(balanced_accuracy_score(y[te], c.predict(b)))
    for k in NAMES: R[k].append(float(np.mean(acc[k])))
    print("  %-8s " % os.path.basename(f).split("_")[0] + " ".join("%s %.3f" % (k[:12], R[k][-1]) for k in NAMES), flush=True)

base = np.array(R["raw_riem"])
print("\n=== GROUP (place-stops) ===")
for k in NAMES:
    a = np.array(R[k]); _, p0 = ttest_1samp(a, 0.5); _, pb = ttest_rel(a, base)
    print("  %-16s %.3f  vs-chance p=%.4f  vs-raw_riem %+.3f (p=%.4f)" % (k, a.mean(), p0, a.mean()-base.mean(), pb))
