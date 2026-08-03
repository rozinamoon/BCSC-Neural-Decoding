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
from pyriemann.tangentspace import TangentSpace

PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
BASE = (-0.2, -0.11)


def load(f, wmax):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    k = ((meta.task == "single-phoneme") & meta.tms_target.str.startswith("control")).values
    bm = (t >= BASE[0]) & (t <= BASE[1]); wm = (t >= 0.0) & (t <= wmax)
    Xk = X[k]
    return (Xk - Xk[:, :, bm].mean(2, keepdims=True))[:, :, wm], meta.phoneme.values[k]


def project(Xtr, Xte, npca):
    pc = PCA(min(npca, len(Xtr) - 2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    f = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    return f(Xtr), f(Xte)


def delay_embed(A, p, tau):
    if p == 1:
        return A
    T = A.shape[2]
    span = (p - 1) * tau
    return np.concatenate([A[:, :, i * tau:T - span + i * tau] for i in range(p)], axis=1)


def unified_cov(A, p, tau, augment, shrink=0.1):
    """Single SPD matrix per trial: delay blocks give dynamics, the constant row gives the
    evoked mean (only if augment=True, i.e. no centring)."""
    E = delay_embed(A, p, tau)
    n, k, T = E.shape
    if augment:
        E = np.concatenate([E, np.ones((n, 1, T))], axis=1)
        k += 1
        C = np.einsum("nit,njt->nij", E, E) / T
    else:
        Ec = E - E.mean(2, keepdims=True)
        C = np.einsum("nit,njt->nij", Ec, Ec) / T
    tr = np.trace(C, axis1=1, axis2=2)[:, None, None] / k
    return (1 - shrink) * C + shrink * tr * np.eye(k)


def tangent(Ctr, Cte):
    ts = TangentSpace(metric="riemann").fit(Ctr)
    return ts.transform(Ctr), ts.transform(Cte)


def concat_baseline(Atr, Ate, K):
    ta, tb, ma, mb = [], [], [], []
    ed = np.linspace(0, Atr.shape[2], K + 1).astype(int)
    for i in range(K):
        sa, sb = Atr[:, :, ed[i]:ed[i+1]], Ate[:, :, ed[i]:ed[i+1]]
        Ca = Covariances("oas").fit_transform(sa)
        Cb = Covariances("oas").fit_transform(sb)
        a, b = tangent(Ca, Cb)
        ta.append(a); tb.append(b); ma.append(sa.mean(2)); mb.append(sb.mean(2))
    ta, tb = np.stack(ta, 1), np.stack(tb, 1)
    ma, mb = np.stack(ma, 1), np.stack(mb, 1)
    bt = [ta.reshape(len(ta), -1), ma.reshape(len(ma), -1)]
    be = [tb.reshape(len(tb), -1), mb.reshape(len(mb), -1)]
    if K > 1:
        bt += [np.diff(ta, axis=1).reshape(len(ta), -1), np.diff(ma, axis=1).reshape(len(ma), -1)]
        be += [np.diff(tb, axis=1).reshape(len(tb), -1), np.diff(mb, axis=1).reshape(len(mb), -1)]
    return bt, be


def std_cat(bt, be):
    sc = [StandardScaler().fit(x) for x in bt]
    return (np.hstack([s.transform(x) for s, x in zip(sc, bt)]),
            np.hstack([s.transform(x) for s, x in zip(sc, be)]))


def ev(Ftr, ytr, Fte, yte):
    m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    m.fit(Ftr, ytr); return balanced_accuracy_score(yte, m.predict(Fte))


FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
CFG = {"place": dict(npca=40, K=2), "ph11": dict(npca=25, K=6)}
NAMES = ["base_0.5", "base_0.8", "U_p1_aug", "U_p2_aug", "U_p3_aug",
         "U_p3_noaug", "U_p2_aug_K2", "U_p2_aug+base"]
RES = {t: {k: [] for k in NAMES} for t in CFG}
print("Unified spatio-temporal SPD matrix vs separate-then-concatenate\n")
print("  delay blocks = dynamics,  constant row = evoked mean,  one manifold point per trial\n")
for f in FILES:
    for tn, cfg in CFG.items():
        acc = {k: [] for k in NAMES}
        for wmax, tag in [(0.5, "base_0.5"), (0.8, "base_0.8")]:
            Xall, ph = load(f, wmax)
            m = np.isin(ph, STOPS) if tn == "place" else np.isin(ph, PHON)
            X = Xall[m]
            y = (np.where(np.isin(ph[m], ["d", "t"]), 1, 0) if tn == "place"
                 else np.array([PHON.index(v) for v in ph[m]]))
            for sd in range(2):
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
                    Atr, Ate = project(X[tr], X[te], cfg["npca"])
                    bt, be = concat_baseline(Atr, Ate, cfg["K"])
                    a, b = std_cat(bt, be)
                    acc[tag].append(ev(a, y[tr], b, y[te]))
        Xall, ph = load(f, 0.8)
        m = np.isin(ph, STOPS) if tn == "place" else np.isin(ph, PHON)
        X = Xall[m]
        y = (np.where(np.isin(ph[m], ["d", "t"]), 1, 0) if tn == "place"
             else np.array([PHON.index(v) for v in ph[m]]))
        dU = 20 if tn == "ph11" else 25
        tau = 30
        for sd in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
                Atr, Ate = project(X[tr], X[te], dU)
                for p, aug, nm in [(1, True, "U_p1_aug"), (2, True, "U_p2_aug"),
                                   (3, True, "U_p3_aug"), (3, False, "U_p3_noaug")]:
                    Ca = unified_cov(Atr, p, tau, aug)
                    Cb = unified_cov(Ate, p, tau, aug)
                    a, b = tangent(Ca, Cb)
                    acc[nm].append(ev(a, y[tr], b, y[te]))
                T = Atr.shape[2]; half = T // 2
                ua, ub = [], []
                for sl in [slice(0, half), slice(half, T)]:
                    Ca = unified_cov(Atr[:, :, sl], 2, tau, True)
                    Cb = unified_cov(Ate[:, :, sl], 2, tau, True)
                    a, b = tangent(Ca, Cb)
                    ua.append(a); ub.append(b)
                a, b = std_cat(ua, ub)
                acc["U_p2_aug_K2"].append(ev(a, y[tr], b, y[te]))
                Atr2, Ate2 = project(X[tr], X[te], cfg["npca"])
                bt, be = concat_baseline(Atr2, Ate2, cfg["K"])
                Ca = unified_cov(Atr, 2, tau, True); Cb = unified_cov(Ate, 2, tau, True)
                ua_, ub_ = tangent(Ca, Cb)
                a, b = std_cat(bt + [ua_], be + [ub_])
                acc["U_p2_aug+base"].append(ev(a, y[tr], b, y[te]))
        for k in NAMES: RES[tn][k].append(float(np.mean(acc[k])))
    print("  %-8s ph11 %s" % (os.path.basename(f).split("_")[0],
          " ".join("%.3f" % RES["ph11"][k][-1] for k in NAMES)), flush=True)

for tn, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tn, ch))
    base = np.array(RES[tn]["base_0.5"])
    for k in NAMES:
        a = np.array(RES[tn][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-16s %.3f  vs-chance p=%.7f  vs-base0.5 %+.3f (p=%.4f) %s" % (
            k, a.mean(), p0, a.mean() - base.mean(), pb,
            "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
