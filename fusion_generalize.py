import warnings
warnings.filterwarnings("ignore")
import glob
import os
import time
import numpy as np
import pandas as pd
from scipy.linalg import logm
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

FILES = [f for f in sorted(glob.glob("data/npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5

def dda(x, tau1=7, tau2=10):
    n, c, T = x.shape; mm = T-tau2-1; o = np.empty((n, c, 4), np.float32)
    for i in range(n):
        for j in range(c):
            u = x[i, j]; ud = u[tau2+1:tau2+1+mm]-u[tau2:tau2+mm]
            a = u[tau2+1-tau1:tau2+1-tau1+mm]; bb = u[1:1+mm]; D = np.stack([a, bb, a*a], 1)
            cf, *_ = np.linalg.lstsq(D, ud, rcond=None); o[i, j] = [cf[0], cf[1], cf[2], float(np.mean((D@cf-ud)**2))]
    return o.reshape(n, -1)

def spd(x, k=15):
    n, c, T = x.shape; p = PCA(k).fit(x.transpose(0, 2, 1).reshape(-1, c))
    xp = np.stack([p.transform(x[i].T).T for i in range(n)], 0)
    iu = np.triu_indices(k); off = iu[0] != iu[1]; F = np.empty((n, len(iu[0])), np.float32)
    for i in range(n):
        s = xp[i]; cov = (s@s.T)/(s.shape[1]-1); cov = 0.85*cov+0.15*np.trace(cov)/k*np.eye(k)
        L = logm(cov).real; v = L[iu].copy(); v[off] *= np.sqrt(2); F[i] = v
    return F

def K(F):
    Fs = StandardScaler().fit_transform(F); return rbf_kernel(Fs, gamma=1.0/(Fs.shape[1]*Fs.var()))

def cv(Klist, y, seed, w):
    a = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        Ktr = sum(w[i]*Klist[i][np.ix_(tr, tr)] for i in range(len(Klist)))
        Kte = sum(w[i]*Klist[i][np.ix_(te, tr)] for i in range(len(Klist)))
        clf = SVC(kernel="precomputed", class_weight="balanced"); clf.fit(Ktr, y[tr])
        a.append(balanced_accuracy_score(y[te], clf.predict(Kte)))
    return float(np.mean(a))

TARGETS = ["category", "manner", "voicing", "place-stops"]
R = {tg: {"V": [], "ADD": []} for tg in TARGETS}
t0 = time.time()
print("Fusion generalization across gross features | %d subj\n" % len(FILES))
print("%-7s | " % "subj" + "  ".join("%s(V/ADD)" % tg[:4] for tg in TARGETS), flush=True)
for idx, f in enumerate(FILES, 1):
    sub = os.path.basename(f).split("_")[0]
    d = np.load(f, allow_pickle=True); X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "category", "phoneme", "manner", "voicing"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    Xs, ds = X[m], df[m].reset_index(drop=True)
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (Xs - Xs[:, :, b].mean(2, keepdims=True))[:, :, w]
    Kv, Kd, Ksp = K(Xb[:, :, ::10].reshape(len(Xb), -1)), K(dda(Xb)), K(spd(Xb))
    ph = ds.phoneme.values
    specs = {
        "category": (np.ones(len(ds), bool), np.where(ds.category.values != "vowels", 1, 0)),
        "manner": (np.isin(ds.manner.values, ["fricative", "stop"]), (ds.manner.values == "fricative").astype(int)),
        "voicing": (np.isin(ds.voicing.values, ["yes", "no"]), (ds.voicing.values == "yes").astype(int)),
        "place-stops": (np.isin(ph, ["b", "p", "d", "t"]), np.where(np.isin(ph, ["d", "t"]), 1, 0)),
    }
    line = []
    for tg in TARGETS:
        mask, yv = specs[tg]; yy = yv[mask]
        Kl = [Kv[np.ix_(mask, mask)], Kd[np.ix_(mask, mask)], Ksp[np.ix_(mask, mask)]]
        v = np.mean([cv([Kl[0]], yy, s, [1.0]) for s in range(5)])
        ad = np.mean([cv(Kl, yy, s, [1/3, 1/3, 1/3]) for s in range(5)])
        R[tg]["V"].append(v); R[tg]["ADD"].append(ad); line.append("%.3f/%.3f" % (v, ad))
    print("%-7s | %s  [%d/%d %.1fm]" % (sub, "  ".join(line), idx, len(FILES), (time.time()-t0)/60), flush=True)

print("\n=== GROUP: fusion(ADD) vs voltage across gross features ===")
for tg in TARGETS:
    V = np.array(R[tg]["V"]); A = np.array(R[tg]["ADD"])
    _, pv = ttest_1samp(V, .5); _, pa = ttest_1samp(A, .5); _, pc = ttest_rel(A, V)
    print("  %-12s V %.3f(p=%.3f)  ADD %.3f(p=%.3f)  ADD-V=%+.3f  paired-p=%.4f  %s  [%d/%d subj ADD>V]" % (
        tg, V.mean(), pv, A.mean(), pa, A.mean()-V.mean(), pc,
        "**FUSION>V**" if A.mean() > V.mean() and pc < 0.05 else "tie", int((A > V).sum()), len(V)))
