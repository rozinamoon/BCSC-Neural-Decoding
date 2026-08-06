"""
Evaluate an evolved Type I substrate on EEG.

Two questions, in order:
  1. Is the substrate operating normally on real EEG?  (survival, saturation, state variance)
  2. Does it carry decodable phoneme information?      (place, 11-phoneme)

Usage:
    python eval_type1.py --params results_eeg/seed0.npy
    python eval_type1.py --params results_eeg/seed0.npy --compare old_seed.npy
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import glob
import os

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import lax, vmap
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann

N, NT, TAU, KAP, DEC, CLIP, EMAX = 32, 10, 8, 1.0, 0.98, 10.0, 4.0
DT_IZH, V_THRESH, I_TONIC, INPUT_SCALE, SYN_DECAY = 0.5, 30.0, 2.95, 30.0, 0.8
P_STD, SUB = 0.3, 10
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]


def unpack(flat):
    i = 0
    W = flat[i:i+N*N].reshape(N, N); i += N*N
    b = flat[i:i+N]; i += N
    ns = N // 4
    w_in = flat[i:i+N] * jnp.zeros(N).at[:ns].set(1.0); i += N
    w_out = flat[i:i+N] * jnp.zeros(N).at[-ns:].set(1.0); i += N
    a = flat[i:i+N]; i += N
    bp = flat[i:i+N]; i += N
    c = flat[i:i+N]; i += N
    d = flat[i:i+N]; i += N
    return W, b, w_in, w_out, a, bp, c, d


def loop(p_seq, W, b, w_in, w_out, a, bp, c, d):
    ns = N // 4

    def step(cr, pt):
        v, u, sp, syn, buf, e = cr
        obs = jnp.concatenate([jnp.full(ns, e), jnp.zeros(N - ns)])
        I_ext = w_in * obs * INPUT_SCALE

        def tick(c2, _):
            vv, uu, ss, sy = c2
            sy_n = SYN_DECAY * sy + ss
            I_syn = W @ sy_n + b + I_ext + I_TONIC
            dv = 0.04 * vv * vv + 5.0 * vv + 140.0 - uu + I_syn
            du = a * (bp * vv - uu)
            v_n = jnp.clip(vv + dv * DT_IZH, -100.0, 1e4)
            u_n = uu + du * DT_IZH
            fired = v_n >= V_THRESH
            v_n = jnp.where(fired, c, v_n)
            u_n = jnp.where(fired, u_n + d, u_n)
            v_n = jnp.clip(v_n, -100.0, V_THRESH)
            f = fired.astype(vv.dtype)
            return (v_n, u_n, f, sy_n), f

        (v, u, sp, syn), all_sp = lax.scan(tick, (v, u, sp, syn), None, length=NT)
        rate = jnp.mean(all_sp, axis=0)
        act = jnp.tanh(jnp.dot(w_out, rate)) * 2.0
        land = buf[0]
        buf = jnp.roll(buf, -1).at[-1].set(act)
        e2 = jnp.clip(DEC * e + pt + KAP * jnp.tanh(land), -CLIP, CLIP)
        return (v, u, sp, syn, buf, e2), (rate, e2)

    init = (jnp.full(N, -65.0), jnp.zeros(N), jnp.zeros(N), jnp.zeros(N),
            jnp.zeros(TAU), 0.0)
    _, out = lax.scan(step, init, p_seq)
    return out


def features(P, flat):
    W, b, w_in, w_out, a, bp, c, d = unpack(jnp.asarray(flat))
    rates, e = vmap(lambda p: loop(p, W, b, w_in, w_out, a, bp, c, d))(jnp.asarray(P))
    R = np.asarray(rates)
    E = np.asarray(e)
    return np.hstack([R[:, ::SUB, :].reshape(len(P), -1), E[:, ::SUB]]), R, E


def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    k = ((meta.task == "single-phoneme") & meta.tms_target.str.startswith("control")).values
    bm = (t >= -0.2) & (t <= -0.11); wm = (t >= 0.0) & (t <= 0.5)
    Xk = X[k]
    return (Xk - Xk[:, :, bm].mean(2, keepdims=True))[:, :, wm], meta.phoneme.values[k]


WIN, NPCA_COV = 50, 10


def scalar_obs(X, mode="geom"):
    """mode='pc1'  : first principal component of the raw signal
       mode='geom' : sliding-window covariance -> tangent space at the Karcher mean -> PC1
                     (unsupervised; matches evolve_geom.py with STRIDE=1)"""
    if mode == "pc1":
        pc = PCA(1).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
        U = np.stack([pc.transform(z.T) for z in X], 0)[:, :, 0]
        return (U / (U.std() + 1e-9) * P_STD).astype(np.float32)
    pc = PCA(NPCA_COV).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
    Z = np.stack([pc.transform(z.T).T for z in X], 0)
    T = Z.shape[2]
    starts = range(0, T - WIN + 1)
    C = np.stack([np.einsum("nkt,nlt->nkl", Z[:, :, s:s + WIN], Z[:, :, s:s + WIN]) / WIN
                  for s in starts], 1)
    d = C.shape[-1]
    C = C + 1e-6 * np.trace(C, axis1=-2, axis2=-1)[..., None, None] / d * np.eye(d)
    flat = C.reshape(-1, d, d)
    M = mean_riemann(flat[::37])
    ts = TangentSpace(metric="riemann")
    ts.reference_ = M
    ts.metric = "riemann"
    V = ts.fit_transform(flat) if not hasattr(ts, "reference_") else None
    from pyriemann.utils.tangentspace import tangent_space as _tsf
    V = _tsf(flat, M).reshape(C.shape[0], C.shape[1], -1)
    p1 = PCA(1).fit(V.reshape(-1, V.shape[2]))
    U = np.stack([p1.transform(v)[:, 0] for v in V], 0)
    return (U / (U.std() + 1e-9) * P_STD).astype(np.float32)


def winriem(Xtr, Xte, K, npca=20):
    pc = PCA(min(npca, len(Xtr)-2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    pr = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = pr(Xtr), pr(Xte)
    ed = np.linspace(0, A.shape[2], K+1).astype(int)
    fa, fb = [], []
    for i in range(K):
        Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
        Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
    return np.hstack(fa), np.hstack(fb)


def ev(Ftr, ytr, Fte, yte):
    m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    m.fit(Ftr, ytr); return balanced_accuracy_score(yte, m.predict(Fte))


def cat(bt, be):
    sc = [StandardScaler().fit(x) for x in bt]
    return (np.hstack([s.transform(x) for s, x in zip(sc, bt)]),
            np.hstack([s.transform(x) for s, x in zip(sc, be)]))


def main(args):
    P_ = {"evolved": np.load(args.params).astype(np.float32)}
    if args.compare and os.path.exists(args.compare):
        P_["old"] = np.load(args.compare).astype(np.float32)
    for e_ in args.extra:
        if os.path.exists(e_):
            P_[os.path.basename(e_).replace(".npy", "")] = np.load(e_).astype(np.float32)
    rng = np.random.default_rng(0)
    P_["shuffled"] = rng.permutation(P_["evolved"])
    print("params: %s" % {k: "|mean| %.3f" % np.abs(v).mean() for k, v in P_.items()})

    files = [f for f in sorted(glob.glob(os.path.join(args.data, "sub-S*_ses-02.npz")))
             if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
    tags = list(P_)
    NAMES = ["winriem"] + ["T1_" + t for t in tags] + ["win+T1_evolved"]
    RES = {tk: {k: [] for k in NAMES} for tk in ["place", "ph11"]}
    DIAG = {t: [] for t in tags}

    print("\nType I substrate evaluation on EEG\n")
    for f in files:
        Xall, ph = load(f)
        tasks = {"place": (Xall[np.isin(ph, STOPS)],
                           np.where(np.isin(ph[np.isin(ph, STOPS)], ["d", "t"]), 1, 0), 2),
                 "ph11": (Xall[np.isin(ph, PHON)],
                          np.array([PHON.index(v) for v in ph[np.isin(ph, PHON)]]), 6)}
        for tn, (X, y, K) in tasks.items():
            P = scalar_obs(X, args.obs)
            F = {}
            for t in tags:
                F[t], R, E = features(P, P_[t])
                if tn == "ph11":
                    DIAG[t].append((float(np.mean(np.abs(E) <= EMAX)),
                                    float(np.mean(np.abs(E) > 9.5)),
                                    float(R.mean(1).var(0).mean()),
                                    float(R.mean())))
            acc = {k: [] for k in NAMES}
            for s in range(2):
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                    Rtr, Rte = winriem(X[tr], X[te], K)
                    acc["winriem"].append(ev(Rtr, y[tr], Rte, y[te]))
                    for t in tags:
                        acc["T1_" + t].append(ev(F[t][tr], y[tr], F[t][te], y[te]))
                    a_, b_ = cat([Rtr, F["evolved"][tr]], [Rte, F["evolved"][te]])
                    acc["win+T1_evolved"].append(ev(a_, y[tr], b_, y[te]))
            for k in NAMES: RES[tn][k].append(float(np.mean(acc[k])))
        print("  %-8s ph11 %s" % (os.path.basename(f).split("_")[0],
              " ".join("%.3f" % RES["ph11"][k][-1] for k in NAMES)), flush=True)

    print("\n=== SUBSTRATE DIAGNOSTIC on real EEG (250-step trials) ===")
    print("  %-10s %10s %10s %11s %9s" % ("params", "survive", "clip_sat", "rate_var", "firing"))
    for t in tags:
        d_ = np.array(DIAG[t]).mean(0)
        print("  %-10s %10.3f %10.4f %11.5f %9.4f" % (t, d_[0], d_[1], d_[2], d_[3]))
    print("  (survive = frac of time |e| <= EMAX=4; evolution reached ~0.62 on 800-step episodes)")

    for tn, ch in [("place", 0.5), ("ph11", 1/11)]:
        print("\n=== %s (chance %.3f) ===" % (tn, ch))
        base = np.array(RES[tn]["winriem"])
        for k in NAMES:
            a_ = np.array(RES[tn][k]); _, p0 = ttest_1samp(a_, ch); _, pb = ttest_rel(a_, base)
            print("  %-16s %.3f  vs-chance p=%.5f  vs-winriem %+.3f (p=%.4f) %s" % (
                k, a_.mean(), p0, a_.mean()-base.mean(), pb,
                "**BEATS**" if pb < 0.05 and a_.mean() > base.mean() else ""))
        if "T1_evolved" in RES[tn] and "T1_shuffled" in RES[tn]:
            g = np.array(RES[tn]["T1_evolved"]); r = np.array(RES[tn]["T1_shuffled"])
            _, pg = ttest_rel(g, r)
            print("  --- evolved vs shuffled: %+.3f (p=%.4f) %s" % (
                g.mean()-r.mean(), pg,
                "EVOLUTION HELPS" if pg < 0.05 and g.mean() > r.mean() else "no effect"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--data", default="npz")
    ap.add_argument("--obs", choices=["geom", "pc1"], default="geom")
    ap.add_argument("--extra", nargs="*", default=[])
    main(ap.parse_args())
