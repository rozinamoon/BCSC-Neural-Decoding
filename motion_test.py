import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances

PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
NPCA = 20


def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    k = ((meta.task == "single-phoneme") & meta.tms_target.str.startswith("control")).values
    bm = (t >= -0.2) & (t <= -0.11); wm = (t >= 0.0) & (t <= 0.5)
    Xk = X[k]
    return (Xk - Xk[:, :, bm].mean(2, keepdims=True))[:, :, wm], meta.phoneme.values[k]


def spd_pow(C, a):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    return (V * (w ** a)[..., None, :]) @ np.swapaxes(V, -1, -2)


def spd_log(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    return (V * np.log(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def triu_vec(S):
    d = S.shape[-1]
    iu = np.triu_indices(d)
    sc = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return S[..., iu[0], iu[1]] * sc


def window_covs(Z, K, overlap=0.0):
    T = Z.shape[2]
    L = int(T / (K - (K - 1) * overlap)) if overlap > 0 else T // K
    stride = max(1, int(L * (1 - overlap)))
    starts = [s for s in range(0, T - L + 1, stride)][:K]
    return [Covariances("oas").fit_transform(Z[:, :, s:s + L]) for s in starts]


def f_position(covs, tr):
    out = []
    for C in covs:
        L = spd_log(C)
        out.append(triu_vec(L - L[tr].mean(0)))
    return np.hstack(out)


def f_motion(covs):
    out = []
    for i in range(len(covs) - 1):
        Aih = spd_pow(covs[i], -0.5)
        out.append(triu_vec(spd_log(np.einsum("nij,njk,nkl->nil", Aih, covs[i + 1], Aih))))
    return np.hstack(out)


def reservoir(Z, seed=0, n_res=200, sub=20):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (n_res, n_res)) * (rng.random((n_res, n_res)) < 0.1)
    W = W / np.max(np.abs(np.linalg.eigvals(W)))
    Win = rng.normal(0, 1, (n_res, Z.shape[1])) * 0.5
    leaks = np.exp(rng.uniform(np.log(0.02), np.log(0.9), n_res))[None, :]
    U = np.transpose(Z, (0, 2, 1))
    U = (U - U.mean((0, 1))) / (U.std((0, 1)) + 1e-9)
    h = np.zeros((len(U), n_res)); out = []
    for ti in range(U.shape[1]):
        h = (1 - leaks) * h + leaks * np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
        if ti % sub == 0: out.append(h.copy())
    return np.stack(out, 1).reshape(len(U), -1)


def ev(Ftr, ytr, Fte, yte):
    m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    m.fit(Ftr, ytr); return balanced_accuracy_score(yte, m.predict(Fte))


def cat(bt, be):
    sc = [StandardScaler().fit(x) for x in bt]
    return (np.hstack([s.transform(x) for s, x in zip(sc, bt)]),
            np.hstack([s.transform(x) for s, x in zip(sc, be)]))


FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
KPOS = {"place": 2, "ph11": 6}
NAMES = ["pos_best", "pos+res", "motion_K4", "motion_K6", "motion_K6_ov",
         "pos+motion_K4", "pos+motion_K6", "pos+motion+res"]
RES = {t: {k: [] for k in NAMES} for t in ["place", "ph11"]}
print("Geodesic displacement on HARD windows (PCA-20 basis, LDA-shrinkage throughout)\n")
for f in FILES:
    Xall, ph = load(f)
    tasks = {"place": (np.isin(ph, STOPS),
                       np.where(np.isin(ph[np.isin(ph, STOPS)], ["d", "t"]), 1, 0)),
             "ph11": (np.isin(ph, PHON),
                      np.array([PHON.index(v) for v in ph[np.isin(ph, PHON)]]))}
    for tn, (m, y) in tasks.items():
        X = Xall[m]
        acc = {k: [] for k in NAMES}
        for sd in range(3):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
                pc = PCA(NPCA).fit(X[tr].transpose(0, 2, 1).reshape(-1, X.shape[1]))
                Z = np.stack([pc.transform(z.T).T for z in X], 0)
                cb = window_covs(Z, KPOS[tn])
                c4 = window_covs(Z, 4)
                c6 = window_covs(Z, 6)
                c6o = window_covs(Z, 6, overlap=0.5)
                P = f_position(cb, tr)
                M4, M6, M6o = f_motion(c4), f_motion(c6), f_motion(c6o)
                R = reservoir(Z)
                acc["pos_best"].append(ev(P[tr], y[tr], P[te], y[te]))
                a, b = cat([P[tr], R[tr]], [P[te], R[te]])
                acc["pos+res"].append(ev(a, y[tr], b, y[te]))
                acc["motion_K4"].append(ev(M4[tr], y[tr], M4[te], y[te]))
                acc["motion_K6"].append(ev(M6[tr], y[tr], M6[te], y[te]))
                acc["motion_K6_ov"].append(ev(M6o[tr], y[tr], M6o[te], y[te]))
                a, b = cat([P[tr], M4[tr]], [P[te], M4[te]])
                acc["pos+motion_K4"].append(ev(a, y[tr], b, y[te]))
                a, b = cat([P[tr], M6[tr]], [P[te], M6[te]])
                acc["pos+motion_K6"].append(ev(a, y[tr], b, y[te]))
                a, b = cat([P[tr], M6[tr], R[tr]], [P[te], M6[te], R[te]])
                acc["pos+motion+res"].append(ev(a, y[tr], b, y[te]))
        for k in NAMES: RES[tn][k].append(float(np.mean(acc[k])))
    print("  %-8s ph11 %s" % (os.path.basename(f).split("_")[0],
          " ".join("%.3f" % RES["ph11"][k][-1] for k in NAMES)), flush=True)

for tn, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tn, ch))
    base = np.array(RES[tn]["pos_best"])
    for k in NAMES:
        a = np.array(RES[tn][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-16s %.3f  vs-chance p=%.7f  vs-pos %+.3f (p=%.4f) %s" % (
            k, a.mean(), p0, a.mean() - base.mean(), pb,
            "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
