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

def prob_model(Ftr, ytr, Fte, classes):
    c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5))
    c.fit(Ftr, ytr)
    P = np.zeros((len(Fte), len(classes)))
    for j, cl in enumerate(c.classes_):
        P[:, list(classes).index(cl)] = c.predict_proba(Fte)[:, j]
    return P

W, Win = make_esn(0)
METHODS = ["winriem", "xdawn", "reservoir", "smooth", "voltage"]
RES = {t: {m: [] for m in METHODS + ["ENSEMBLE"]} for t in ["place", "ph11"]}
COMP = {t: [] for t in ["place", "ph11"]}
print("ENSEMBLE of complementary assumptions | do different priors make different errors?\n")
for f in FILES:
    Xall, ph = load(f)
    k1 = np.isin(ph, ["b", "p", "d", "t"]); k2 = np.isin(ph, PHON)
    tasks = {"place": (Xall[k1], np.where(np.isin(ph[k1], ["d", "t"]), 1, 0), 2),
             "ph11": (Xall[k2], np.array([PHON.index(v) for v in ph[k2]]), 6)}
    for tname, (X, y, K) in tasks.items():
        classes = np.unique(y)
        correct = {m: [] for m in METHODS}
        accs = {m: [] for m in METHODS + ["ENSEMBLE"]}
        for s in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                P = {}
                a, b = winriem_feats(X[tr], X[te], K)
                P["winriem"] = prob_model(a, y[tr], b, classes)
                xd = Xdawn(nfilter=4, estimator="oas").fit(X[tr], y[tr])
                Ztr, Zte = xd.transform(X[tr]), xd.transform(X[te])
                P["xdawn"] = prob_model(Ztr[:, :, ::10].reshape(len(tr), -1), y[tr], Zte[:, :, ::10].reshape(len(te), -1), classes)
                a, b = esn_feats(X[tr], X[te], W, Win)
                P["reservoir"] = prob_model(a, y[tr], b, classes)
                Str, Ste = smooth(X[tr]), smooth(X[te])
                P["smooth"] = prob_model(Str[:, :, ::10].reshape(len(tr), -1), y[tr], Ste[:, :, ::10].reshape(len(te), -1), classes)
                P["voltage"] = prob_model(X[tr][:, :, ::10].reshape(len(tr), -1), y[tr], X[te][:, :, ::10].reshape(len(te), -1), classes)
                for m in METHODS:
                    pred = classes[np.argmax(P[m], 1)]
                    accs[m].append(balanced_accuracy_score(y[te], pred))
                    if s == 0: correct[m].append((pred == y[te]).astype(int))
                ens = classes[np.argmax(np.mean([P[m] for m in METHODS], 0), 1)]
                accs["ENSEMBLE"].append(balanced_accuracy_score(y[te], ens))
        for m in METHODS + ["ENSEMBLE"]: RES[tname][m].append(float(np.mean(accs[m])))
        C = {m: np.concatenate(correct[m]) for m in METHODS}
        cc = [np.corrcoef(C[a], C[b])[0, 1] for i, a in enumerate(METHODS) for b in METHODS[i+1:]]
        anyc = np.mean(np.any(np.stack([C[m] for m in METHODS]), 0))
        COMP[tname].append((np.nanmean(cc), anyc, np.mean([C[m].mean() for m in METHODS])))
    print("  %s done" % os.path.basename(f).split("_")[0], flush=True)

for tname, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tname, ch))
    best = max(METHODS, key=lambda m: np.mean(RES[tname][m]))
    for m in METHODS + ["ENSEMBLE"]:
        a = np.array(RES[tname][m]); _, p0 = ttest_1samp(a, ch)
        _, pb = ttest_rel(a, np.array(RES[tname][best])) if m != best else (0, np.nan)
        print("  %-10s %.3f  vs-chance p=%.6f  vs-best(%s) %+.3f (p=%.4f)" % (
            m, a.mean(), p0, best, a.mean()-np.mean(RES[tname][best]), pb))
    cc, anyc, meanc = np.array(COMP[tname]).mean(0)
    print("  --- complementarity: mean pairwise error-corr %.3f | any-method-correct %.3f vs mean-single %.3f" % (cc, anyc, meanc))
