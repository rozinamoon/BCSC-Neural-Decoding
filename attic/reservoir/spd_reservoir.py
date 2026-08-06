import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
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
D_SPD, LAM, STEP = 10, 0.25, 10

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & df.tms_target.str.startswith("control")).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df.phoneme.values[m]

def spd_pow(P, a):
    w, V = np.linalg.eigh(P)
    w = np.clip(w, 1e-10, None)
    return np.einsum("nij,nj,nkj->nik", V, w ** a, V)

def geodesic(P, S, lam):
    Ph = spd_pow(P, 0.5); Pmh = spd_pow(P, -0.5)
    M = np.einsum("nij,njk,nkl->nil", Pmh, S, Pmh)
    Mp = spd_pow(M, lam)
    return np.einsum("nij,njk,nkl->nil", Ph, Mp, Ph)

def spd_reservoir(U, n_read=6):
    n, T, d = U.shape
    P = np.tile(np.eye(d), (n, 1, 1))
    reads = []
    read_at = set(np.linspace(STEP, T - 1, n_read).astype(int))
    for ti in range(0, T, STEP):
        seg = U[:, ti:ti + STEP, :]
        S = np.einsum("ntd,nte->nde", seg, seg) / max(1, seg.shape[1])
        S = S + 0.15 * np.trace(S, axis1=1, axis2=2)[:, None, None] / d * np.eye(d)
        P = geodesic(P, S, LAM)
        if any(abs(ti - r) < STEP for r in read_at):
            reads.append(P.copy())
    return reads[:n_read]

def orth_esn(U, n_res=200, leak=0.3, seed=0, sub=20):
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1, (n_res, n_res))
    Q, _ = np.linalg.qr(A)
    Win = rng.normal(0, 1, (n_res, U.shape[2])) * 0.5
    n, T, _ = U.shape
    h = np.zeros((n, n_res)); out = []
    for ti in range(T):
        h = (1 - leak) * h + leak * np.tanh(h @ Q.T + U[:, ti, :] @ Win.T)
        if ti % sub == 0: out.append(h.copy())
    return np.stack(out, 1).reshape(n, -1)

def std_esn(U, n_res=200, leak=0.3, seed=0, sub=20):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (n_res, n_res)) * (rng.random((n_res, n_res)) < 0.1)
    W = W / np.max(np.abs(np.linalg.eigvals(W)))
    Win = rng.normal(0, 1, (n_res, U.shape[2])) * 0.5
    n, T, _ = U.shape
    h = np.zeros((n, n_res)); out = []
    for ti in range(T):
        h = (1 - leak) * h + leak * np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
        if ti % sub == 0: out.append(h.copy())
    return np.stack(out, 1).reshape(n, -1)

def winriem(Xtr, Xte, K, npca=20):
    pc = PCA(npca).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    proj = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = proj(Xtr), proj(Xte)
    T = A.shape[2]; ed = np.linspace(0, T, K + 1).astype(int)
    fa, fb = [], []
    for i in range(K):
        Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
        Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
    return np.hstack(fa), np.hstack(fb)

def ev(Ftr, ytr, Fte, yte):
    c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    c.fit(Ftr, ytr); return balanced_accuracy_score(yte, c.predict(Fte))

NAMES = ["winriem", "SPD_RESERVOIR", "SPD_RES+winriem", "orth_esn", "std_esn"]
RES = {t: {k: [] for k in NAMES} for t in ["place", "ph11"]}
print("Riemannian INSIDE the reservoir: state = SPD matrix, update = geodesic step\n")
for f in FILES:
    Xall, ph = load(f)
    k1 = np.isin(ph, ["b", "p", "d", "t"]); k2 = np.isin(ph, PHON)
    tasks = {"place": (Xall[k1], np.where(np.isin(ph[k1], ["d", "t"]), 1, 0), 2),
             "ph11": (Xall[k2], np.array([PHON.index(v) for v in ph[k2]]), 6)}
    for tname, (X, y, K) in tasks.items():
        pcg = PCA(D_SPD).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
        U = np.stack([pcg.transform(z.T) for z in X], 0)
        U = (U - U.mean((0, 1))) / (U.std((0, 1)) + 1e-9)
        SPD_STATES = spd_reservoir(U)
        E_orth, E_std = orth_esn(U), std_esn(U)
        acc = {k: [] for k in NAMES}
        for s in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                Rtr, Rte = winriem(X[tr], X[te], K)
                acc["winriem"].append(ev(Rtr, y[tr], Rte, y[te]))
                fa, fb = [], []
                for P in SPD_STATES:
                    ts = TangentSpace(metric="riemann").fit(P[tr])
                    fa.append(ts.transform(P[tr])); fb.append(ts.transform(P[te]))
                fa, fb = np.hstack(fa), np.hstack(fb)
                acc["SPD_RESERVOIR"].append(ev(fa, y[tr], fb, y[te]))
                s1 = StandardScaler().fit(Rtr); s2 = StandardScaler().fit(fa)
                a = np.hstack([s1.transform(Rtr), s2.transform(fa)])
                b = np.hstack([s1.transform(Rte), s2.transform(fb)])
                acc["SPD_RES+winriem"].append(ev(a, y[tr], b, y[te]))
                acc["orth_esn"].append(ev(E_orth[tr], y[tr], E_orth[te], y[te]))
                acc["std_esn"].append(ev(E_std[tr], y[tr], E_std[te], y[te]))
        for k in NAMES: RES[tname][k].append(float(np.mean(acc[k])))
    print("  %-8s place %s | ph11 %s" % (os.path.basename(f).split("_")[0],
          " ".join("%.3f" % RES["place"][k][-1] for k in NAMES),
          " ".join("%.3f" % RES["ph11"][k][-1] for k in NAMES)), flush=True)

for tname, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tname, ch))
    base = np.array(RES[tname]["winriem"])
    for k in NAMES:
        a = np.array(RES[tname][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-18s %.3f  vs-chance p=%.6f  vs-winriem %+.3f (p=%.4f) %s" % (
            k, a.mean(), p0, a.mean()-base.mean(), pb, "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
