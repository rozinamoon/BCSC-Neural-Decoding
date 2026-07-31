import warnings
warnings.filterwarnings("ignore")
import argparse
import glob
import os
import time
import numpy as np
from multiprocessing import Pool
from scipy.stats import ttest_rel, t as tdist
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from pyriemann.spatialfilters import Xdawn
import fastriem as fr
import fastlda as fl
import csd as csdmod

PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
CHANCE = {"place": 0.5, "ph11": 1 / 11}
NF = {"place": 6, "ph11": 1}
DPCA = 20
SPACES = ("raw", "car", "csd")
WINS = {"full": (0.0, 1.01, 6), "late": (0.2, 1.01, 5)}
REPS = ("riem", "proto")
GRID = [(s, w, r) for s in SPACES for w in WINS for r in REPS]
GMAX = 0.75


def load(path):
    d = np.load(path, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    k = (d["task"].astype(str) == "single_phoneme") & (d["tms_target"].astype(str) == "control")
    X = X[k] - X[k][:, :, t <= 0.02].mean(2, keepdims=True)
    return X, t, d["phoneme1"].astype(str)[k], np.asarray(d["channel_positions_xyz"], float)


def spatial(X, mode, L):
    if mode == "raw":
        return X
    if mode == "car":
        return X - X.mean(1, keepdims=True)
    return csdmod.apply_csd(X, L)


def blocks_riem(Xtr, Xte, K, M0):
    pc = PCA(min(DPCA, len(Xtr) - 2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    pr = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    return _wins(pr(Xtr), pr(Xte), K, M0)


def blocks_proto(Xtr, ytr, Xte, nf, K, M0):
    xd = Xdawn(nfilter=nf, estimator="lwf").fit(Xtr, ytr)
    V, cls = xd.filters_, np.unique(ytr)
    P = np.concatenate([V[i * nf:(i + 1) * nf] @ Xtr[ytr == c].mean(0)
                        for i, c in enumerate(cls)], 0)
    fa, fb = xd.transform(Xtr), xd.transform(Xte)
    Za = np.concatenate([np.repeat(P[None], len(fa), 0), fa], 1)
    Zb = np.concatenate([np.repeat(P[None], len(fb), 0), fb], 1)
    return _wins(Za, Zb, K, M0)


def _wins(A, B, K, M0):
    ed = np.linspace(0, A.shape[2], K + 1).astype(int)
    if A.shape[1] / (A.shape[2] // K) > GMAX:
        raise ValueError("gamma too high")
    a, b, Ms = [], [], []
    for i in range(K):
        Ca, Cb = fr.cov_oas(A[:, :, ed[i]:ed[i + 1]]), fr.cov_oas(B[:, :, ed[i]:ed[i + 1]])
        M = fr.riemann_mean(Ca, init=None if M0 is None else M0[i])
        Ms.append(M)
        a.append(fr.tangent(Ca, M))
        b.append(fr.tangent(Cb, M))
    return np.concatenate(a, 1), np.concatenate(b, 1), Ms


def clf(a, ytr, b, yte):
    sc = StandardScaler().fit(a)
    co, it, cl = fl.fit(sc.transform(a), ytr)
    return balanced_accuracy_score(yte, fl.predict(sc.transform(b), co, it, cl))


def evaluate(X, t, y, L, nf, nseed, perm):
    rng = np.random.default_rng(999) if perm else None
    yy = rng.permutation(y) if perm else y
    acc = {g: [] for g in GRID}
    dims = {}
    M0 = {g: None for g in GRID}
    for sd in range(nseed):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, yy):
            for sp in SPACES:
                Xs = spatial(X, sp, L)
                for wn, (t0, t1, K) in WINS.items():
                    msk = (t >= t0) & (t < t1)
                    A, B = Xs[tr][:, :, msk], Xs[te][:, :, msk]
                    for rp in REPS:
                        g = (sp, wn, rp)
                        try:
                            if rp == "riem":
                                a, b, M0[g] = blocks_riem(A, B, K, M0[g])
                            else:
                                a, b, M0[g] = blocks_proto(A, yy[tr], B, nf, K, M0[g])
                            dims[g] = a.shape[1]
                            acc[g].append(clf(a, yy[tr], b, yy[te]))
                        except Exception:
                            M0[g] = None
    return {g: (float(np.mean(v)) if v else float("nan")) for g, v in acc.items()}, dims


def one(job):
    path, tn, nseed = job
    t_start = time.time()
    X, t, ph, pos = load(path)
    L = csdmod.csd_operator(pos)
    keep = STOPS if tn == "place" else PHON
    m = np.isin(ph, keep)
    p = ph[m]
    y = (np.where(np.isin(p, ["d", "t"]), 1, 0) if tn == "place"
         else np.array([PHON.index(v) for v in p]))
    real, dims = evaluate(X[m], t, y, L, NF[tn], nseed, False)
    perm, _ = evaluate(X[m], t, y, L, NF[tn], 1, True)
    return os.path.basename(path)[:3], tn, real, perm, dims, time.time() - t_start


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tasks", nargs="*", default=["place", "ph11"])
    ap.add_argument("--subjects", type=int, default=0)
    A = ap.parse_args()
    F = sorted(glob.glob("npz2/*.npz"))
    if A.subjects:
        F = F[:A.subjects]
    jobs = [(f, tk, A.seeds) for f in F for tk in A.tasks]
    print("FACTORIAL | space{raw,car,csd} x window{full,late} x rep{riem,proto}")
    print("%d jobs | %d configs | seeds %d | + permutation control\n"
          % (len(jobs), len(GRID), A.seeds), flush=True)
    R, P, D = {}, {}, {}
    with Pool(A.pool) as pool:
        for i, (sub, tn, r, pm, dims, el) in enumerate(pool.imap_unordered(one, jobs), 1):
            R[(sub, tn)] = r
            P[(sub, tn)] = pm
            D.update(dims)
            v = {k: x for k, x in r.items() if not np.isnan(x)}
            pv = [x for x in pm.values() if not np.isnan(x)]
            print("  [%2d/%2d] %s %-5s best=%.3f %s | permMax=%.3f  %.0fs"
                  % (i, len(jobs), sub, tn, max(v.values()), max(v, key=v.get),
                     max(pv) if pv else float("nan"), el), flush=True)

    def ci(d):
        m, s, n = d.mean(), d.std(ddof=1), len(d)
        h = tdist.ppf(.975, n - 1) * s / np.sqrt(n)
        return m, m - h, m + h

    subs = sorted({s for s, _ in R})
    for tn in A.tasks:
        print("\n=== %s (chance %.3f, n=%d) ===" % (tn, CHANCE[tn], len(subs)))
        tab = {}
        for g in GRID:
            v = np.array([R[(s, tn)][g] for s in subs])
            if not np.isnan(v).any():
                tab[g] = v
        print("  %-22s %-6s %s" % ("config", "dim", "acc"))
        for g in sorted(tab):
            print("  %-6s %-5s %-8s %-6d %.4f" % (g[0], g[1], g[2], D.get(g, 0), tab[g].mean()))
        pm = np.array([max(x for x in P[(s, tn)].values() if not np.isnan(x)) for s in subs])
        print("  permutation ceiling (max over configs): %.4f" % pm.mean())
        best = max(tab, key=lambda k: tab[k].mean())
        print("  BEST %s = %.4f   selection-corrected %.4f   (%d/%d above chance)"
              % (best, tab[best].mean(), tab[best].mean() - (pm.mean() - CHANCE[tn]),
                 int((tab[best] > CHANCE[tn]).sum()), len(subs)))
        for w in WINS:
            for r in REPS:
                for a, b in [("csd", "car"), ("car", "raw"), ("csd", "raw")]:
                    if (a, w, r) in tab and (b, w, r) in tab:
                        d = tab[(a, w, r)] - tab[(b, w, r)]
                        m, lo, hi = ci(d)
                        print("    %-5s %-6s %s-%s: %+.4f [%+.4f,%+.4f] p=%.4f"
                              % (w, r, a, b, m, lo, hi,
                                 ttest_rel(tab[(a, w, r)], tab[(b, w, r)]).pvalue))
