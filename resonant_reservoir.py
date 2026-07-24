import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from scipy.signal import butter, filtfilt
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
WMIN, WMAX, SF = 0.0, 0.5, 500.0
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
N_RES, D_IN, SUB = 200, 20, 20

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & df.tms_target.str.startswith("control")).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df.phoneme.values[m]

def base_w(seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.1)
    W = W / np.max(np.abs(np.linalg.eigvals(W)))
    return W, rng.normal(0, 1, (N_RES, D_IN)) * 0.5

def esn_uniform(U, W, Win, leak=0.3):
    n, T, _ = U.shape; h = np.zeros((n, N_RES)); out = []
    for ti in range(T):
        h = (1-leak)*h + leak*np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
        if ti % SUB == 0: out.append(h.copy())
    return np.stack(out, 1).reshape(n, -1)

def esn_multitau(U, W, Win, seed=0):
    rng = np.random.default_rng(seed + 77)
    leaks = np.exp(rng.uniform(np.log(0.02), np.log(0.9), N_RES))[None, :]
    n, T, _ = U.shape; h = np.zeros((n, N_RES)); out = []
    for ti in range(T):
        h = (1-leaks)*h + leaks*np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
        if ti % SUB == 0: out.append(h.copy())
    return np.stack(out, 1).reshape(n, -1)

def esn_resonant(U, W, Win, seed=0, fmin=1.0, fmax=45.0, damp=0.90, gain=0.5):
    rng = np.random.default_rng(seed + 99)
    freqs = np.exp(rng.uniform(np.log(fmin), np.log(fmax), N_RES))
    om = 2*np.pi*freqs/SF
    a1 = (2*damp*np.cos(om))[None, :]; a2 = (damp**2)[None, :] if np.ndim(damp) else damp**2
    n, T, _ = U.shape
    x0 = np.zeros((n, N_RES)); x1 = np.zeros((n, N_RES)); out = []
    for ti in range(T):
        drive = np.tanh(x0 @ W.T + U[:, ti, :] @ Win.T) * gain
        x_new = a1*x0 - a2*x1 + drive
        x_new = np.clip(x_new, -50, 50)
        x1, x0 = x0, x_new
        if ti % SUB == 0: out.append(x0.copy())
    S = np.stack(out, 1)
    S = S / (S.std((0, 1), keepdims=True) + 1e-9)
    return np.concatenate([S.reshape(n, -1), np.abs(S).mean(1)], 1)

def winriem(Xtr, Xte, K, npca=20):
    pc = PCA(npca).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    proj = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = proj(Xtr), proj(Xte)
    T = A.shape[2]; ed = np.linspace(0, T, K+1).astype(int)
    fa, fb = [], []
    for i in range(K):
        Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
        Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
        t_ = TangentSpace(metric="riemann").fit(Ca)
        fa.append(t_.transform(Ca)); fb.append(t_.transform(Cb))
    return np.hstack(fa), np.hstack(fb)

def ev(Ftr, ytr, Fte, yte):
    c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    c.fit(Ftr, ytr); return balanced_accuracy_score(yte, c.predict(Fte))

def cat(a, b, c, d):
    s1 = StandardScaler().fit(a); s2 = StandardScaler().fit(c)
    return np.hstack([s1.transform(a), s2.transform(c)]), np.hstack([s1.transform(b), s2.transform(d)])

BANDS = [(1, 4), (4, 8), (8, 15), (15, 30), (30, 60)]

def band_timecourse(X3, npca=12):
    from sklearn.decomposition import PCA as _P
    pc = _P(npca).fit(X3.transpose(0, 2, 1).reshape(-1, X3.shape[1]))
    Z = np.stack([pc.transform(z.T).T for z in X3], 0)
    outs = []
    for lo, hi in BANDS:
        b, a = butter(4, [lo/(SF/2), hi/(SF/2)], btype="band")
        outs.append(filtfilt(b, a, Z, axis=2)[:, :, ::10].reshape(len(Z), -1))
    return np.hstack(outs)

NAMES = ["winriem", "esn_uniform", "MULTITAU", "BAND_TIMECOURSE", "win+MULTITAU", "win+BANDTC", "win+BANDTC+MULTI"]
RES = {t: {k: [] for k in NAMES} for t in ["place", "ph11"]}
W, Win = base_w(0)
print("FREQUENCY-SELECTIVE NEURONS inside the reservoir (multi-timescale / resonant band-pass)\n")
for f in FILES:
    Xall, ph = load(f)
    k1 = np.isin(ph, ["b", "p", "d", "t"]); k2 = np.isin(ph, PHON)
    tasks = {"place": (Xall[k1], np.where(np.isin(ph[k1], ["d", "t"]), 1, 0), 2),
             "ph11": (Xall[k2], np.array([PHON.index(v) for v in ph[k2]]), 6)}
    for tname, (X, y, K) in tasks.items():
        pc = PCA(D_IN).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
        U = np.stack([pc.transform(z.T) for z in X], 0)
        U = (U - U.mean((0, 1))) / (U.std((0, 1)) + 1e-9)
        F_uni, F_mt, F_rs = esn_uniform(U, W, Win), esn_multitau(U, W, Win), band_timecourse(X)
        acc = {k: [] for k in NAMES}
        for s in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                Rtr, Rte = winriem(X[tr], X[te], K)
                acc["winriem"].append(ev(Rtr, y[tr], Rte, y[te]))
                acc["esn_uniform"].append(ev(F_uni[tr], y[tr], F_uni[te], y[te]))
                acc["MULTITAU"].append(ev(F_mt[tr], y[tr], F_mt[te], y[te]))
                acc["BAND_TIMECOURSE"].append(ev(F_rs[tr], y[tr], F_rs[te], y[te]))
                a, b = cat(Rtr, Rte, F_mt[tr], F_mt[te]); acc["win+MULTITAU"].append(ev(a, y[tr], b, y[te]))
                a, b = cat(Rtr, Rte, F_rs[tr], F_rs[te]); acc["win+BANDTC"].append(ev(a, y[tr], b, y[te]))
                s3 = StandardScaler().fit(F_mt[tr])
                a2 = np.hstack([a, s3.transform(F_mt[tr])]); b2 = np.hstack([b, s3.transform(F_mt[te])])
                acc["win+BANDTC+MULTI"].append(ev(a2, y[tr], b2, y[te]))
        for k in NAMES: RES[tname][k].append(float(np.mean(acc[k])))
    print("  %-8s place %s | ph11 %s" % (os.path.basename(f).split("_")[0],
          " ".join("%.3f" % RES["place"][k][-1] for k in NAMES),
          " ".join("%.3f" % RES["ph11"][k][-1] for k in NAMES)), flush=True)

for tname, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tname, ch))
    base = np.array(RES[tname]["winriem"])
    for k in NAMES:
        a = np.array(RES[tname][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-16s %.3f  vs-chance p=%.7f  vs-winriem %+.3f (p=%.4f) %s" % (
            k, a.mean(), p0, a.mean()-base.mean(), pb, "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
