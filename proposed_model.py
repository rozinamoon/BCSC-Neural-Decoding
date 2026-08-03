import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space as tsf

PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
SIGMAS = [15.0, 40.0, 90.0]
N_CENTRES, DELTA = 4, 25


def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    k = ((meta.task == "single-phoneme") & meta.tms_target.str.startswith("control")).values
    bm = (t >= -0.2) & (t <= -0.11); wm = (t >= 0.0) & (t <= 0.5)
    Xk = X[k]
    return (Xk - Xk[:, :, bm].mean(2, keepdims=True))[:, :, wm], meta.phoneme.values[k]


def electrode_positions(path="npz/sub-S04_ses-02.npz"):
    import mne
    mne.set_log_level("ERROR")
    d = np.load(path, allow_pickle=True)
    ch = [str(c) for c in d["ch_names"]]
    info = mne.create_info(ch, 500.0, "eeg")
    info.set_montage("standard_1020", match_case=False, on_missing="ignore")
    pos = np.array([info["chs"][i]["loc"][:3] for i in range(len(ch))])
    ok = np.isfinite(pos).all(1) & (np.abs(pos).sum(1) > 0)
    return pos, ok, ch


def laplacian_basis(pos, ok, d):
    P = pos.copy()
    P[~ok] = P[ok].mean(0)
    D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    h2 = np.median(D2[D2 > 0])
    A = np.exp(-D2 / (2 * h2))
    np.fill_diagonal(A, 0.0)
    L = np.diag(A.sum(1)) - A
    w, V = eigh(L)
    return V[:, :d]


def gaussian_covs(Z, sigmas, n_centres):
    n, d, T = Z.shape
    centres = np.linspace(0.15 * T, 0.85 * T, n_centres)
    tt = np.arange(T)
    out = {}
    for s in sigmas:
        for ci, c0 in enumerate(centres):
            w = np.exp(-0.5 * ((tt - c0) / s) ** 2)
            w = w / w.sum()
            Zw = Z * np.sqrt(w)[None, None, :]
            C = np.einsum("ndt,net->nde", Zw, Zw)
            C = C + 1e-6 * np.trace(C, axis1=1, axis2=2)[:, None, None] / d * np.eye(d)
            out[(s, ci)] = C
    return out, centres


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


def feats_position(covs, keys, ref_idx):
    """Log-Euclidean tangent space: reference = exp(mean(log C)) over the training fold.
    Closed form, so ~50x faster than the Karcher mean with near-identical discriminability."""
    out = []
    for k in keys:
        C = covs[k]
        L = spd_log(C)
        Mlog = L[ref_idx].mean(0)
        out.append(triu_vec(L - Mlog))
    return np.hstack(out)


def feats_motion(covs, sig, centres_n):
    out = []
    for s in sig:
        for ci in range(centres_n - 1):
            A, B = covs[(s, ci)], covs[(s, ci + 1)]
            Aih = spd_pow(A, -0.5)
            out.append(triu_vec(spd_log(np.einsum("nij,njk,nkl->nil", Aih, B, Aih))))
    return np.hstack(out)


def winriem(Xtr, Xte, K, npca=20):
    pc = PCA(min(npca, len(Xtr) - 2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    pr = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = pr(Xtr), pr(Xte)
    ed = np.linspace(0, A.shape[2], K + 1).astype(int)
    fa, fb = [], []
    for i in range(K):
        Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
        Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
    return np.hstack(fa), np.hstack(fb)


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
    m = (make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-1, 5, 7)))
         if Ftr.shape[1] > 2000
         else make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")))
    m.fit(Ftr, ytr); return balanced_accuracy_score(yte, m.predict(Fte))


def cat(bt, be):
    sc = [StandardScaler().fit(x) for x in bt]
    return (np.hstack([s.transform(x) for s, x in zip(sc, bt)]),
            np.hstack([s.transform(x) for s, x in zip(sc, be)]))


def delta_hyperbolicity(pos, ok, n_sample=400, seed=0):
    P = pos[ok]
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    rng = np.random.default_rng(seed)
    n = len(P)
    vals = []
    for _ in range(n_sample):
        w, x, y, z = rng.choice(n, 4, replace=False)
        s = sorted([D[w, x] + D[y, z], D[w, y] + D[x, z], D[w, z] + D[x, y]])
        vals.append((s[2] - s[1]) / 2.0)
    return float(np.mean(vals)), float(np.mean(D))


FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
POS, OK, CH = electrode_positions()
dlt, dmean = delta_hyperbolicity(POS, OK)
print("electrode geometry: %d/%d channels have positions" % (OK.sum(), len(CH)))
print("  delta-hyperbolicity %.4f   mean distance %.4f   ratio %.3f" % (dlt, dmean, dlt / dmean))
print("  (ratio -> 0 = tree-like, hyperbolic fits;  large = not tree-like)\n")

NAMES = ["A_current_K6", "A_current+res", "B_gauss_pca", "B_gauss_lap",
         "C_gauss_lap+motion", "D_motion_only", "E_B_plus_res", "F_lap40"]
RES = {t: {k: [] for k in NAMES} for t in ["place", "ph11"]}
for f in FILES:
    Xall, ph = load(f)
    tasks = {"place": (np.isin(ph, STOPS),
                       np.where(np.isin(ph[np.isin(ph, STOPS)], ["d", "t"]), 1, 0), 2),
             "ph11": (np.isin(ph, PHON),
                      np.array([PHON.index(v) for v in ph[np.isin(ph, PHON)]]), 6)}
    for tn, (m, y, K) in tasks.items():
        X = Xall[m]
        U20 = laplacian_basis(POS, OK, 20)
        U40 = laplacian_basis(POS, OK, 40)
        Zl20 = np.einsum("ck,nct->nkt", U20, X)
        Zl40 = np.einsum("ck,nct->nkt", U40, X)
        pc = PCA(20).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
        Zp = np.stack([pc.transform(z.T).T for z in X], 0)
        cov_p, _ = gaussian_covs(Zp, SIGMAS, N_CENTRES)
        cov_l, _ = gaussian_covs(Zl20, SIGMAS, N_CENTRES)
        cov_l40, _ = gaussian_covs(Zl40, SIGMAS, N_CENTRES)
        keys = [(s, ci) for s in SIGMAS for ci in range(N_CENTRES)]
        res_l = reservoir(Zl20)
        acc = {k: [] for k in NAMES}
        for sd in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
                Rtr, Rte = winriem(X[tr], X[te], K)
                acc["A_current_K6"].append(ev(Rtr, y[tr], Rte, y[te]))
                a, b = cat([Rtr, res_l[tr]], [Rte, res_l[te]])
                acc["A_current+res"].append(ev(a, y[tr], b, y[te]))
                Fp = feats_position(cov_p, keys, tr)
                acc["B_gauss_pca"].append(ev(Fp[tr], y[tr], Fp[te], y[te]))
                Fl = feats_position(cov_l, keys, tr)
                acc["B_gauss_lap"].append(ev(Fl[tr], y[tr], Fl[te], y[te]))
                Fm = feats_motion(cov_l, SIGMAS, N_CENTRES)
                a, b = cat([Fl[tr], Fm[tr]], [Fl[te], Fm[te]])
                acc["C_gauss_lap+motion"].append(ev(a, y[tr], b, y[te]))
                acc["D_motion_only"].append(ev(Fm[tr], y[tr], Fm[te], y[te]))
                a, b = cat([Fl[tr], res_l[tr]], [Fl[te], res_l[te]])
                acc["E_B_plus_res"].append(ev(a, y[tr], b, y[te]))
                F40 = feats_position(cov_l40, keys, tr)
                acc["F_lap40"].append(ev(F40[tr], y[tr], F40[te], y[te]))
        for k in NAMES: RES[tn][k].append(float(np.mean(acc[k])))
    print("  %-8s ph11 %s" % (os.path.basename(f).split("_")[0],
          " ".join("%.3f" % RES["ph11"][k][-1] for k in NAMES)), flush=True)

for tn, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tn, ch))
    base = np.array(RES[tn]["A_current_K6"])
    for k in NAMES:
        a = np.array(RES[tn][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-20s %.3f  vs-chance p=%.6f  vs-A %+.3f (p=%.4f) %s" % (
            k, a.mean(), p0, a.mean() - base.mean(), pb,
            "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
