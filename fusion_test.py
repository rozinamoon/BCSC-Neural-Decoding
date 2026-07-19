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

def dda(x: np.ndarray, tau1: int = 7, tau2: int = 10) -> np.ndarray:
    n, c, T = x.shape; mm = T - tau2 - 1; o = np.empty((n, c, 4), np.float32)
    for i in range(n):
        for j in range(c):
            u = x[i, j]; ud = u[tau2+1:tau2+1+mm] - u[tau2:tau2+mm]
            a = u[tau2+1-tau1:tau2+1-tau1+mm]; bb = u[1:1+mm]; D = np.stack([a, bb, a*a], 1)
            cf, *_ = np.linalg.lstsq(D, ud, rcond=None)
            o[i, j] = [cf[0], cf[1], cf[2], float(np.mean((D@cf-ud)**2))]
    return o.reshape(n, -1)

def spd(x: np.ndarray, k: int = 15) -> np.ndarray:
    n, c, T = x.shape
    p = PCA(k).fit(x.transpose(0, 2, 1).reshape(-1, c))
    xp = np.stack([p.transform(x[i].T).T for i in range(n)], 0)
    iu = np.triu_indices(k); off = iu[0] != iu[1]; F = np.empty((n, len(iu[0])), np.float32)
    for i in range(n):
        s = xp[i]; cov = (s@s.T)/(s.shape[1]-1); cov = 0.85*cov + 0.15*np.trace(cov)/k*np.eye(k)
        L = logm(cov).real; v = L[iu].copy(); v[off] *= np.sqrt(2); F[i] = v
    return F

def K(F: np.ndarray) -> np.ndarray:
    Fs = StandardScaler().fit_transform(F)
    return rbf_kernel(Fs, gamma=1.0/(Fs.shape[1]*Fs.var()))

def cvK(Kmat: np.ndarray, y: np.ndarray, seed: int) -> float:
    a = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        clf = SVC(kernel="precomputed", class_weight="balanced")
        clf.fit(Kmat[np.ix_(tr, tr)], y[tr])
        a.append(balanced_accuracy_score(y[te], clf.predict(Kmat[np.ix_(te, tr)])))
    return float(np.mean(a))

def oof_correct(Kmat: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    pred = np.zeros(len(y), int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        clf = SVC(kernel="precomputed", class_weight="balanced"); clf.fit(Kmat[np.ix_(tr, tr)], y[tr])
        pred[te] = clf.predict(Kmat[np.ix_(te, tr)])
    return (pred == y).astype(int)

R = {c: {"V": [], "D": [], "S": [], "VD": [], "VS": [], "ADD": []} for c in ["cat", "pl"]}
ERRCORR = []
t0 = time.time()
print("FUSION test: voltage vs voltage+DDA+SPD (additive) | %d subj\n" % len(FILES))
print("%-7s | place-stops: V   D   S   V+D  V+S  ADD | errcorr(V,D)/(V,S)/(D,S)" % "subj")
for idx, f in enumerate(FILES, 1):
    sub = os.path.basename(f).split("_")[0]
    d = np.load(f, allow_pickle=True); X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "category", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    Xs, ds = X[m], df[m].reset_index(drop=True)
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (Xs - Xs[:, :, b].mean(2, keepdims=True))[:, :, w]
    Fv = Xb[:, :, ::10].reshape(len(Xb), -1); Fd = dda(Xb); Fs = spd(Xb)
    Kv, Kd, Ks = K(Fv), K(Fd), K(Fs)
    ph = ds.phoneme.values
    for cond, mask, yv in [
        ("cat", np.ones(len(ds), bool), np.where(ds.category.values != "vowels", 1, 0)),
        ("pl", np.isin(ph, ["b", "p", "d", "t"]), np.where(np.isin(ph, ["d", "t"]), 1, 0))]:
        yy = yv[mask]
        kv, kd, ks = Kv[np.ix_(mask, mask)], Kd[np.ix_(mask, mask)], Ks[np.ix_(mask, mask)]
        kadd = (kv + kd + ks)/3; kvd = (kv + kd)/2; kvs = (kv + ks)/2
        sc = lambda Km: float(np.mean([cvK(Km, yy, s) for s in range(5)]))
        R[cond]["V"].append(sc(kv)); R[cond]["D"].append(sc(kd)); R[cond]["S"].append(sc(ks))
        R[cond]["VD"].append(sc(kvd)); R[cond]["VS"].append(sc(kvs)); R[cond]["ADD"].append(sc(kadd))
    m2 = np.isin(ph, ["b", "p", "d", "t"]); yy = np.where(np.isin(ph[m2], ["d", "t"]), 1, 0)
    cv = oof_correct(Kv[np.ix_(m2, m2)], yy, 0); cd = oof_correct(Kd[np.ix_(m2, m2)], yy, 0); cs = oof_correct(Ks[np.ix_(m2, m2)], yy, 0)
    ec = (np.corrcoef(cv, cd)[0, 1], np.corrcoef(cv, cs)[0, 1], np.corrcoef(cd, cs)[0, 1])
    ERRCORR.append(ec)
    r = R["pl"]
    print("%-7s | %.3f %.3f %.3f %.3f %.3f %.3f | %+.2f %+.2f %+.2f  [%d/%d %.1fm]" % (
        sub, r["V"][-1], r["D"][-1], r["S"][-1], r["VD"][-1], r["VS"][-1], r["ADD"][-1],
        ec[0], ec[1], ec[2], idx, len(FILES), (time.time()-t0)/60), flush=True)

print("\n=== GROUP: does fusion beat voltage-alone? (paired-t, ADD vs V) ===")
for cond, lbl, ch in [("cat", "category", .5), ("pl", "place-stops", .5)]:
    V = np.array(R[cond]["V"]); ADD = np.array(R[cond]["ADD"])
    VD = np.array(R[cond]["VD"]); VS = np.array(R[cond]["VS"])
    _, pv = ttest_1samp(V, ch); _, pa = ttest_1samp(ADD, ch); _, pc = ttest_rel(ADD, V)
    _, pvd = ttest_rel(VD, V); _, pvs = ttest_rel(VS, V)
    print("  %-12s V %.3f(p=%.3f)  ADD %.3f(p=%.3f)  ADD-V=%+.3f p=%.3f %s | V+D-V=%+.3f(p=%.2f) V+S-V=%+.3f(p=%.2f)" % (
        lbl, V.mean(), pv, ADD.mean(), pa, ADD.mean()-V.mean(), pc,
        "FUSION>V" if ADD.mean() > V.mean() and pc < 0.05 else "tie",
        VD.mean()-V.mean(), pvd, VS.mean()-V.mean(), pvs))
ec = np.array(ERRCORR)
print("\nerror-correlation (place-stops, lower=more complementary):")
print("  V-D %.2f  V-S %.2f  D-S %.2f  (independent errors -> fusion should help)" % (ec[:,0].mean(), ec[:,1].mean(), ec[:,2].mean()))
