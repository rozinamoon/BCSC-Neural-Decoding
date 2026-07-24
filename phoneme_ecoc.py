import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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
CODE = np.array([
    [1, 0, 0, 1, 0], [1, 1, 1, 1, 0], [1, 1, 0, 1, 0], [1, 0, 1, 1, 1], [1, 0, 0, 1, 1],
    [0, 1, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 1, 1, 1], [0, 1, 1, 0, 0], [0, 1, 1, 1, 0]])
D_IN, N_RES, LEAK = 20, 200, 0.3

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w]
    ph = df.phoneme.values[m]
    k = np.isin(ph, PHON)
    return Xb[k], np.array([PHON.index(v) for v in ph[k]])

def win_riem(Str, Ste, K=4, npca=20):
    p = PCA(npca).fit(Str.transpose(0, 2, 1).reshape(-1, Str.shape[1]))
    pr = lambda Z: np.stack([p.transform(z.T).T for z in Z], 0)
    Str, Ste = pr(Str), pr(Ste)
    T = Str.shape[2]; ed = np.linspace(0, T, K+1).astype(int)
    ftr, fte = [], []
    for i in range(K):
        Ca = Covariances("oas").fit_transform(Str[:, :, ed[i]:ed[i+1]])
        Cb = Covariances("oas").fit_transform(Ste[:, :, ed[i]:ed[i+1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        ftr.append(ts.transform(Ca)); fte.append(ts.transform(Cb))
    return np.hstack(ftr), np.hstack(fte)

def make_esn(seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.1)
    W = W / np.max(np.abs(np.linalg.eigvals(W)))
    return W, rng.normal(0, 1, (N_RES, D_IN)) * 0.5

def esn_feats(Xtr, Xte, W, Win):
    p = PCA(D_IN).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    def run(Z):
        U = np.stack([p.transform(z.T) for z in Z], 0)
        n, T, _ = U.shape; h = np.zeros((n, N_RES)); out = []
        for ti in range(T):
            h = (1-LEAK)*h + LEAK*np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
            if ti % 20 == 0: out.append(h.copy())
        return np.stack(out, 1).reshape(n, -1)
    return run(Xtr), run(Xte)

def lin(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5))
def svm(): return make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale", class_weight="balanced", C=2.0))

def ecoc(Ftr, ytr, Fte):
    probs = []
    for j in range(CODE.shape[1]):
        bj = CODE[ytr, j]
        if len(np.unique(bj)) < 2:
            probs.append(np.full(len(Fte), 0.5)); continue
        c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5))
        c.fit(Ftr, bj); probs.append(c.predict_proba(Fte)[:, 1])
    P = np.clip(np.stack(probs, 1), 1e-6, 1-1e-6)
    ll = P[:, None, :] * CODE[None, :, :] + (1-P)[:, None, :] * (1-CODE)[None, :, :]
    return np.argmax(np.log(ll).sum(2), 1)

NAMES = ["winriem_lin", "winriem_svm", "WINRIEM_ECOC", "res_lin", "res_svm", "RES_ECOC"]
R = {k: [] for k in NAMES}
W, Win = make_esn(0)
print("11-phoneme: SVM readout + articulatory-feature ECOC | chance %.3f\n" % (1/11))
for f in FILES:
    X, y = load(f)
    acc = {k: [] for k in NAMES}
    for s in range(3):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            a, b = win_riem(X[tr], X[te])
            c = lin(); c.fit(a, y[tr]); acc["winriem_lin"].append(balanced_accuracy_score(y[te], c.predict(b)))
            c = svm(); c.fit(a, y[tr]); acc["winriem_svm"].append(balanced_accuracy_score(y[te], c.predict(b)))
            acc["WINRIEM_ECOC"].append(balanced_accuracy_score(y[te], ecoc(a, y[tr], b)))
            a2, b2 = esn_feats(X[tr], X[te], W, Win)
            c = lin(); c.fit(a2, y[tr]); acc["res_lin"].append(balanced_accuracy_score(y[te], c.predict(b2)))
            c = svm(); c.fit(a2, y[tr]); acc["res_svm"].append(balanced_accuracy_score(y[te], c.predict(b2)))
            acc["RES_ECOC"].append(balanced_accuracy_score(y[te], ecoc(a2, y[tr], b2)))
    for k in NAMES: R[k].append(float(np.mean(acc[k])))
    print("  %-8s " % os.path.basename(f).split("_")[0] + " ".join("%s %.3f" % (k[:11], R[k][-1]) for k in NAMES), flush=True)

base = np.array(R["winriem_lin"])
print("\n=== GROUP (11-phoneme, chance 0.091) ===")
for k in NAMES:
    a = np.array(R[k]); _, p0 = ttest_1samp(a, 1/11); _, pb = ttest_rel(a, base)
    print("  %-14s %.3f  vs-chance p=%.6f  vs-winriem_lin %+.3f (p=%.4f)" % (k, a.mean(), p0, a.mean()-base.mean(), pb))
