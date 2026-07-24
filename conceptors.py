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
from pyriemann.tangentspace import TangentSpace

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
D_IN, LEAK = 20, 0.3
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df.phoneme.values[m]

def make_w(n_res, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (n_res, n_res)) * (rng.random((n_res, n_res)) < 0.1)
    W = W / np.max(np.abs(np.linalg.eigvals(W)))
    return W, rng.normal(0, 1, (n_res, D_IN)) * 0.5

def states(U, W, Win, n_res):
    n, T, _ = U.shape
    h = np.zeros((n, n_res)); out = []
    for ti in range(T):
        h = (1-LEAK)*h + LEAK*np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
        out.append(h.copy())
    return np.stack(out, 1)

def prep(Xtr, Xte):
    p = PCA(D_IN).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    f = lambda Z: np.stack([p.transform(z.T) for z in Z], 0)
    A, B = f(Xtr), f(Xte)
    mu, sd = A.mean((0, 1)), A.std((0, 1)) + 1e-9
    return (A - mu)/sd, (B - mu)/sd

def conceptor(R, alpha):
    return R @ np.linalg.inv(R + alpha**-2 * np.eye(R.shape[0]))

def class_conceptor_clf(Str, ytr, Ste, alpha=8.0):
    n_res = Str.shape[2]
    Cs = {}
    for k in np.unique(ytr):
        Z = Str[ytr == k].reshape(-1, n_res)
        R = Z.T @ Z / len(Z)
        Cs[k] = conceptor(R, alpha)
    ks = sorted(Cs)
    ev = np.zeros((len(Ste), len(ks)))
    for i, z in enumerate(Ste):
        nrm = (z * z).sum(1) + 1e-12
        for j, k in enumerate(ks):
            ev[i, j] = np.mean(np.einsum("ti,ij,tj->t", z, Cs[k], z) / nrm)
    return np.array(ks)[np.argmax(ev, 1)]

def trial_conceptor_feats(Str, Ste, alpha=8.0):
    n_res = Str.shape[2]
    def cmat(S):
        out = []
        for z in S:
            R = z.T @ z / len(z)
            C = conceptor(R, alpha)
            C = 0.5*(C + C.T) + 1e-6*np.eye(n_res)
            out.append(C)
        return np.stack(out, 0)
    A, B = cmat(Str), cmat(Ste)
    ts = TangentSpace(metric="riemann").fit(A)
    return ts.transform(A), ts.transform(B)

def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5))

W200, Win200 = make_w(200, 0)
W30, Win30 = make_w(30, 0)
NAMES = ["res_linear", "CLASS_CONCEPTOR", "TRIAL_CONCEPTOR_RIEM"]
RES = {t: {k: [] for k in NAMES} for t in ["place", "ph11"]}
print("CONCEPTORS: geometric prior inside the reservoir | place & 11-phoneme\n")
for f in FILES:
    Xall, ph = load(f)
    k1 = np.isin(ph, ["b", "p", "d", "t"]); k2 = np.isin(ph, PHON)
    tasks = {"place": (Xall[k1], np.where(np.isin(ph[k1], ["d", "t"]), 1, 0)),
             "ph11": (Xall[k2], np.array([PHON.index(v) for v in ph[k2]]))}
    for tname, (X, y) in tasks.items():
        acc = {k: [] for k in NAMES}
        for s in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                Utr, Ute = prep(X[tr], X[te])
                S1tr, S1te = states(Utr, W200, Win200, 200), states(Ute, W200, Win200, 200)
                F1, F2 = S1tr[:, ::20].reshape(len(tr), -1), S1te[:, ::20].reshape(len(te), -1)
                c = lr(); c.fit(F1, y[tr]); acc["res_linear"].append(balanced_accuracy_score(y[te], c.predict(F2)))
                pred = class_conceptor_clf(S1tr, y[tr], S1te)
                acc["CLASS_CONCEPTOR"].append(balanced_accuracy_score(y[te], pred))
                S2tr, S2te = states(Utr, W30, Win30, 30), states(Ute, W30, Win30, 30)
                a, b = trial_conceptor_feats(S2tr, S2te)
                c = lr(); c.fit(a, y[tr]); acc["TRIAL_CONCEPTOR_RIEM"].append(balanced_accuracy_score(y[te], c.predict(b)))
        for k in NAMES: RES[tname][k].append(float(np.mean(acc[k])))
    print("  %-8s place %s | ph11 %s" % (os.path.basename(f).split("_")[0],
          " ".join("%.3f" % RES["place"][k][-1] for k in NAMES),
          " ".join("%.3f" % RES["ph11"][k][-1] for k in NAMES)), flush=True)

for tname, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tname, ch))
    base = np.array(RES[tname]["res_linear"])
    for k in NAMES:
        a = np.array(RES[tname][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-22s %.3f  vs-chance p=%.5f  vs-res_linear %+.3f (p=%.4f)" % (k, a.mean(), p0, a.mean()-base.mean(), pb))
