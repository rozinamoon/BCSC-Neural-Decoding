import warnings
warnings.filterwarnings("ignore")
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

T1 = "/Users/siwonyun/Work/programming/type1_learning/results"
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
N = 32
NT, TAU, KAP, ENV_DECAY, CLIP = 10, 8, 1.0, 0.98, 10.0
DT_IZH, V_THRESH, I_TONIC, INPUT_SCALE, SYN_DECAY = 0.5, 30.0, 2.95, 30.0, 0.8
P_STD = 0.3
SUB = 10


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


def closed_loop_rates(p_seq, W, b, w_in, w_out, a, bp, c, d):
    ns = N // 4
    v0 = jnp.full(N, -65.0); u0 = jnp.zeros(N)
    sp0 = jnp.zeros(N); sy0 = jnp.zeros(N)

    def step(cr, pt):
        v, u, sp, syn, buf, e = cr
        obs_exp = jnp.concatenate([jnp.full(ns, e), jnp.zeros(N - ns)])
        I_ext = w_in * obs_exp * INPUT_SCALE

        def tick(c2, _):
            vv, uu, ss, sy = c2
            sy_n = SYN_DECAY * sy + ss
            I_syn = W @ sy_n + b + I_ext + I_TONIC
            dv = 0.04 * vv * vv + 5.0 * vv + 140.0 - uu + I_syn
            du = a * (bp * vv - uu)
            v_n = vv + dv * DT_IZH
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
        e2 = jnp.clip(ENV_DECAY * e + pt + KAP * jnp.tanh(land), -CLIP, CLIP)
        return (v, u, sp, syn, buf, e2), (rate, e2)

    init = (v0, u0, sp0, sy0, jnp.zeros(TAU), 0.0)
    _, (rates, e_seq) = lax.scan(step, init, p_seq)
    return rates, e_seq


def type1_features(P, flat, open_loop=False):
    W, b, w_in, w_out, a, bp, c, d = unpack(jnp.asarray(flat))
    if open_loop:
        w_out = w_out * 0.0
    run = lambda p: closed_loop_rates(p, W, b, w_in, w_out, a, bp, c, d)
    rates, e_seq = vmap(run)(jnp.asarray(P))
    R = np.asarray(rates)[:, ::SUB, :].reshape(len(P), -1)
    E = np.asarray(e_seq)[:, ::SUB]
    return np.hstack([R, E])


def scalar_obs(X, n_pc=1):
    pc = PCA(n_pc).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
    U = np.stack([pc.transform(z.T) for z in X], 0)
    U = U / (U.std((0, 1), keepdims=True) + 1e-9) * P_STD
    return U.astype(np.float32)


def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    k = ((meta.task == "single-phoneme") & meta.tms_target.str.startswith("control")).values
    bm = (t >= -0.2) & (t <= -0.11); wm = (t >= 0.0) & (t <= 0.5)
    Xk = X[k]
    return (Xk - Xk[:, :, bm].mean(2, keepdims=True))[:, :, wm], meta.phoneme.values[k]


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
    c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    c.fit(Ftr, ytr); return balanced_accuracy_score(yte, c.predict(Fte))


def cat(bt, be):
    sc = [StandardScaler().fit(b) for b in bt]
    return (np.hstack([s.transform(b) for s, b in zip(sc, bt)]),
            np.hstack([s.transform(b) for s, b in zip(sc, be)]))


PARAMS = {}
_ga = np.load(os.path.join(T1, "seeds", "seed0.npy")).astype(np.float32)
PARAMS["GA"] = _ga
_rng = np.random.default_rng(0)
PARAMS["Random"] = _rng.permutation(_ga)
print("params:", {k: (v.shape, float(np.abs(v).mean())) for k, v in PARAMS.items()})

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
NAMES = ["winriem", "T1_GA_closed", "T1_GA_open", "T1_Random_closed",
         "win+T1_GA_closed", "win+T1_Random_closed"]
RES = {t: {k: [] for k in NAMES} for t in ["place", "ph11"]}
DIAG = []
print("\nType I CLOSED LOOP (e-dot = -gamma e + p + kappa tanh(a_r(t-tau)))\n")
for f in FILES:
    Xall, ph = load(f)
    tasks = {"place": (Xall[np.isin(ph, STOPS)],
                       np.where(np.isin(ph[np.isin(ph, STOPS)], ["d", "t"]), 1, 0), 2),
             "ph11": (Xall[np.isin(ph, PHON)],
                      np.array([PHON.index(v) for v in ph[np.isin(ph, PHON)]]), 6)}
    for tn, (X, y, K) in tasks.items():
        P = scalar_obs(X)[:, :, 0]
        F = {"T1_GA_closed": type1_features(P, PARAMS["GA"]),
             "T1_GA_open": type1_features(P, PARAMS["GA"], open_loop=True),
             "T1_Random_closed": type1_features(P, PARAMS["Random"])}
        if tn == "ph11":
            r = F["T1_GA_closed"][:, :-len(F["T1_GA_closed"][0]) // 33]
            DIAG.append((float(np.mean(r > 0)), float(np.std(F["T1_GA_closed"]))))
        acc = {k: [] for k in NAMES}
        for s in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                Rtr, Rte = winriem(X[tr], X[te], K)
                acc["winriem"].append(ev(Rtr, y[tr], Rte, y[te]))
                for nm in ["T1_GA_closed", "T1_GA_open", "T1_Random_closed"]:
                    acc[nm].append(ev(F[nm][tr], y[tr], F[nm][te], y[te]))
                for nm, key in [("win+T1_GA_closed", "T1_GA_closed"),
                                ("win+T1_Random_closed", "T1_Random_closed")]:
                    a, b = cat([Rtr, F[key][tr]], [Rte, F[key][te]])
                    acc[nm].append(ev(a, y[tr], b, y[te]))
        for k in NAMES: RES[tn][k].append(float(np.mean(acc[k])))
    print("  %-8s ph11 %s" % (os.path.basename(f).split("_")[0],
          " ".join("%.3f" % RES["ph11"][k][-1] for k in NAMES)), flush=True)

print("\n  substrate diagnostic: mean firing fraction %.3f, feature std %.3f" % (
    np.mean([d[0] for d in DIAG]), np.mean([d[1] for d in DIAG])))
for tn, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tn, ch))
    base = np.array(RES[tn]["winriem"])
    for k in NAMES:
        a = np.array(RES[tn][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-21s %.3f  vs-chance p=%.5f  vs-winriem %+.3f (p=%.4f) %s" % (
            k, a.mean(), p0, a.mean()-base.mean(), pb,
            "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
    g, r = np.array(RES[tn]["T1_GA_closed"]), np.array(RES[tn]["T1_Random_closed"])
    o = np.array(RES[tn]["T1_GA_open"])
    _, pg = ttest_rel(g, r); _, po = ttest_rel(g, o)
    print("  --- GA vs Random: %+.3f (p=%.4f) | closed vs open loop: %+.3f (p=%.4f)" % (
        g.mean()-r.mean(), pg, g.mean()-o.mean(), po))
