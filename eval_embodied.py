"""
Evaluate embodied Type I substrates (evolve_embodied.py genomes) on EEG.

Critical: the substrate must run in the SAME environment it evolved in, or it operates outside
the regime it was selected for (an earlier evaluation failed exactly this way -- e saturated at
the CLIP floor and carried no information). So this reproduces:
    geometric K-channel observation, hotspot envelope, silent-agent amplitude calibration,
    localized/uniform gate, K-channel regulation with the L2 budget.

kreg is inferred from the genome size: |g| = N^2 + (6 + kreg) * N + N.

    python eval_embodied.py --params result_eeg/o8r8_s0.npy --kobs 8 --embodied 1
    python eval_embodied.py --params result_eeg/o8r8_s0.npy --kobs 8 --embodied 1 \
        --extra result_eeg/o8r8u_s0.npy result_eeg/o4r4_s0.npy
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
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space as tsf

import evolve_embodied as EV

N, EMAX = EV.N, EV.EMAX
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
WIN, NPCA_COV, SUB = 50, 10, 5


def infer_kreg(size: int, n: int = N) -> int:
    return (size - n * n - 7 * n) // n


def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    k = ((meta.task == "single-phoneme") & meta.tms_target.str.startswith("control")).values
    bm = (t >= -0.2) & (t <= -0.11); wm = (t >= 0.0) & (t <= 0.5)
    Xk = X[k]
    return (Xk - Xk[:, :, bm].mean(2, keepdims=True))[:, :, wm], meta.phoneme.values[k]


def geom_channels(X, kobs):
    pc = PCA(NPCA_COV).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
    Z = np.stack([pc.transform(z.T).T for z in X], 0)
    T = Z.shape[2]
    C = np.stack([np.einsum("nkt,nlt->nkl", Z[:, :, s:s+WIN], Z[:, :, s:s+WIN]) / WIN
                  for s in range(0, T - WIN + 1)], 1)
    d = C.shape[-1]
    C = C + 1e-6 * np.trace(C, axis1=-2, axis2=-1)[..., None, None] / d * np.eye(d)
    flat = C.reshape(-1, d, d)
    V = tsf(flat, mean_riemann(flat[::37])).reshape(C.shape[0], C.shape[1], -1)
    pk = PCA(kobs).fit(V.reshape(-1, V.shape[2]))
    return np.stack([pk.transform(v) for v in V], 0)


def prepare_env(P, kreg, target_frac=0.35, seed=0):
    """Reproduce the evolution environment: rms-normalise -> hotspot -> silent-agent calibration."""
    n, T, k = P.shape
    flat = P.reshape(-1, k)
    flat = flat / (np.sqrt(np.mean(flat.std(0) ** 2)) + 1e-9) * EV.P_STD
    flat = EV.hotspot(flat)
    EV.KREG_G[0] = kreg
    amp = EV.calibrate(flat, T, target_frac, seed)
    flat = flat / (flat.std() + 1e-9) * amp
    return flat.reshape(n, T, k)


def rollout(P, flat, kreg, embodied, budget=1.0):
    W, b, w_in, w_out, a, bp, c, d, w_e = EV.split_genome(jnp.asarray(flat), N, kreg)
    k = P.shape[2]
    centres = jnp.arange(k, dtype=jnp.float32) - (k - 1) / 2.0

    def one(p):
        def step(cr, pt):
            v, u, sp, syn, buf_r, buf_e, e, q = cr
            I_ext = w_in * EV.expand_obs(e) * EV.INPUT_SCALE

            def tick(c2, _):
                vv, uu, ss, sy = c2
                sy_n = EV.SYN_DECAY * sy + ss
                I_syn = W @ sy_n + b + I_ext + EV.I_TONIC
                dv = 0.04 * vv * vv + 5.0 * vv + 140.0 - uu + I_syn
                du = a * (bp * vv - uu)
                v_n = jnp.clip(vv + dv * EV.DT_IZH, -100.0, 1e4)
                u_n = uu + du * EV.DT_IZH
                fired = v_n >= EV.V_THRESH
                v_n = jnp.where(fired, c, v_n)
                u_n = jnp.where(fired, u_n + d, u_n)
                v_n = jnp.clip(v_n, -100.0, EV.V_THRESH)
                f = fired.astype(vv.dtype)
                return (v_n, u_n, f, sy_n), f

            (v, u, sp, syn), all_sp = lax.scan(tick, (v, u, sp, syn), None,
                                               length=EV.NT, unroll=EV.NT)
            rate = jnp.mean(all_sp, axis=0)
            g = jnp.tanh(w_out @ rate)
            a_r = g * (budget / jnp.maximum(1.0, jnp.linalg.norm(g)))
            a_e = jnp.tanh(jnp.dot(w_e, rate)) * 2.0
            land_r, land_e = buf_r[0], buf_e[0]
            buf_r = jnp.roll(buf_r, -1, axis=0).at[-1].set(a_r)
            buf_e = jnp.roll(buf_e, -1).at[-1].set(a_e)
            q2 = jnp.clip(q + EV.V_SCALE * land_e, -EV.Q_BOUND, EV.Q_BOUND)
            gate = (jnp.exp(-0.5 * ((q2 - centres) / EV.SHELTER_W) ** 2) if embodied
                    else jnp.ones(k) / jnp.sqrt(k))
            drive = jnp.tanh(land_r[0]) * jnp.ones(k) if kreg == 1 else land_r[:k]
            e2 = jnp.clip(EV.DEC * e + pt + EV.KAP * gate * drive, -EV.CLIP, EV.CLIP)
            return (v, u, sp, syn, buf_r, buf_e, e2, q2), (rate, e2, q2)

        init = (jnp.full(N, -65.0), jnp.zeros(N), jnp.zeros(N), jnp.zeros(N),
                jnp.zeros((EV.TAU, kreg)), jnp.zeros(EV.TAU), jnp.zeros(k), 0.0)
        _, out = lax.scan(step, init, p)
        return out

    rates, e, q = vmap(one)(jnp.asarray(P))
    R, E, Q = np.asarray(rates), np.asarray(e), np.asarray(q)
    F = np.hstack([R[:, ::SUB, :].reshape(len(P), -1),
                   E[:, ::SUB, :].reshape(len(P), -1),
                   Q[:, ::SUB]])
    return F, E


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


def ev(F, y, seeds=2):
    acc = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(F, y):
            m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
            m.fit(F[tr], y[tr]); acc.append(balanced_accuracy_score(y[te], m.predict(F[te])))
    return float(np.mean(acc))


def main(args):
    files = [f for f in sorted(glob.glob(os.path.join(args.data, "sub-S*_ses-02.npz")))
             if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
    subs = {}
    for pth in [args.params] + args.extra:
        if not os.path.exists(pth):
            continue
        g = np.load(pth).astype(np.float32)
        base = os.path.basename(pth).replace(".npy", "")
        lg = pth.replace(".npy", "_log.json")
        kobs = args.kobs
        if os.path.exists(lg):
            import json
            kobs = int(json.load(open(lg))["args"].get("kobs", args.kobs))
        subs[base] = (g, infer_kreg(g.size), kobs)
    rng = np.random.default_rng(0)
    v0 = list(subs.values())[0]
    subs["shuffled"] = (rng.permutation(v0[0]), v0[1], v0[2])
    print("substrates: %s" % {k: "kreg=%d kobs=%d" % (v[1], v[2]) for k, v in subs.items()})

    tags = list(subs)
    NAMES = ["winriem"] + ["T1_" + t for t in tags] + ["win+T1_best"]
    RES = {t: {k: [] for k in NAMES} for t in ["place", "ph11"]}
    DIAG = {t: [] for t in tags}
    print("\nEmbodied Type I evaluation | kobs=%d embodied=%d\n" % (args.kobs, args.embodied))
    for f in files:
        Xall, ph = load(f)
        tasks = {"place": (np.isin(ph, STOPS),
                           np.where(np.isin(ph[np.isin(ph, STOPS)], ["d", "t"]), 1, 0), 2),
                 "ph11": (np.isin(ph, PHON),
                          np.array([PHON.index(v) for v in ph[np.isin(ph, PHON)]]), 6)}
        for tn, (m, y, K) in tasks.items():
            X = Xall[m]
            envs = {}
            for kk in sorted({v[2] for v in subs.values()}):
                envs[kk] = prepare_env(geom_channels(X, kk), 1, seed=0)
            F = {}
            for t in tags:
                gen, kreg, kobs = subs[t]
                F[t], E = rollout(envs[kobs], gen, kreg, bool(args.embodied))
                if tn == "ph11":
                    DIAG[t].append((float(np.mean(np.max(np.abs(E), 2) <= EMAX)),
                                    float(np.mean(np.abs(E) > 9.5)),
                                    float(F[t].var(0).mean())))
            acc = {k: [] for k in NAMES}
            for s in range(2):
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                    Rtr, Rte = winriem(X[tr], X[te], K)
                    acc["winriem"].append(ev_fold(Rtr, y[tr], Rte, y[te]))
                    for t in tags:
                        acc["T1_" + t].append(ev_fold(F[t][tr], y[tr], F[t][te], y[te]))
                    b = tags[0]
                    s1 = StandardScaler().fit(Rtr); s2 = StandardScaler().fit(F[b][tr])
                    A = np.hstack([s1.transform(Rtr), s2.transform(F[b][tr])])
                    B = np.hstack([s1.transform(Rte), s2.transform(F[b][te])])
                    acc["win+T1_best"].append(ev_fold(A, y[tr], B, y[te]))
            for k in NAMES: RES[tn][k].append(float(np.mean(acc[k])))
        print("  %-8s ph11 %s" % (os.path.basename(f).split("_")[0],
              " ".join("%.3f" % RES["ph11"][k][-1] for k in NAMES)), flush=True)

    print("\n=== SUBSTRATE DIAGNOSTIC (real EEG) ===")
    print("  %-14s %9s %10s %10s" % ("substrate", "alive", "clip_sat", "feat_var"))
    for t in tags:
        d_ = np.array(DIAG[t]).mean(0)
        print("  %-14s %9.3f %10.4f %10.5f" % (t, d_[0], d_[1], d_[2]))

    for tn, ch in [("place", 0.5), ("ph11", 1/11)]:
        print("\n=== %s (chance %.3f) ===" % (tn, ch))
        base = np.array(RES[tn]["winriem"])
        for k in NAMES:
            a = np.array(RES[tn][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
            print("  %-18s %.3f  vs-chance p=%.5f  vs-winriem %+.3f (p=%.4f) %s" % (
                k, a.mean(), p0, a.mean()-base.mean(), pb,
                "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
        best = tags[0]
        g = np.array(RES[tn]["T1_" + best]); r = np.array(RES[tn]["T1_shuffled"])
        _, pg = ttest_rel(g, r)
        print("  --- %s vs shuffled: %+.3f (p=%.4f) %s" % (
            best, g.mean()-r.mean(), pg,
            "EVOLUTION HELPS" if pg < 0.05 and g.mean() > r.mean() else "no effect"))


def ev_fold(Ftr, ytr, Fte, yte):
    m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    m.fit(Ftr, ytr)
    return balanced_accuracy_score(yte, m.predict(Fte))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--kobs", type=int, default=8)
    ap.add_argument("--embodied", type=int, default=1)
    ap.add_argument("--data", default="npz")
    main(ap.parse_args())
