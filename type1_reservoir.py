import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import lax
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

jax.config.update("jax_enable_x64", False)

T1 = "/Users/siwonyun/Work/programming/type1_learning/results"
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
N_NEUR, N_TICKS, SUB = 32, 4, 20
DT_IZH, V_THRESH, I_TONIC, INPUT_SCALE, SYN_DECAY = 0.5, 30.0, 2.95, 30.0, 0.8


def unpack(flat: jnp.ndarray, N: int):
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


def izh_scan(I_seq: jnp.ndarray, W, b, a, bp, c, d):
    n_tr = I_seq.shape[0]
    v0 = jnp.full((n_tr, N_NEUR), -65.0)
    u0 = jnp.zeros((n_tr, N_NEUR))
    sp0 = jnp.zeros((n_tr, N_NEUR))
    sy0 = jnp.zeros((n_tr, N_NEUR))

    def outer(carry, I_ext):
        def tick(cr, _):
            v, u, sp, syn = cr
            syn_new = SYN_DECAY * syn + sp
            I_syn = syn_new @ W.T + b + I_ext + I_TONIC
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I_syn
            du = a * (bp * v - u)
            v_new = v + dv * DT_IZH
            u_new = u + du * DT_IZH
            fired = v_new >= V_THRESH
            v_new = jnp.where(fired, c, v_new)
            u_new = jnp.where(fired, u_new + d, u_new)
            v_new = jnp.clip(v_new, -100.0, V_THRESH)
            return (v_new, u_new, fired.astype(v.dtype), syn_new), fired.astype(v.dtype)
        carry, all_sp = lax.scan(tick, carry, None, length=N_TICKS)
        return carry, jnp.mean(all_sp, axis=0)

    _, rates = lax.scan(outer, (v0, u0, sp0, sy0), jnp.transpose(I_seq, (1, 0, 2)))
    return jnp.transpose(rates, (1, 0, 2))


def type1_features(U: np.ndarray, flat: np.ndarray) -> np.ndarray:
    W, b, w_in, w_out, a, bp, c, d = unpack(jnp.asarray(flat), N_NEUR)
    ns = N_NEUR // 4
    n, T, k = U.shape
    inj = np.zeros((n, T, N_NEUR), np.float32)
    inj[:, :, :ns] = U[:, :, :ns] * INPUT_SCALE
    R = np.asarray(izh_scan(jnp.asarray(inj), W, b, a, bp, c, d))
    return R[:, ::SUB, :].reshape(n, -1)


def esn_features(U: np.ndarray, seed: int = 0, n_res: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (n_res, n_res)) * (rng.random((n_res, n_res)) < 0.2)
    W = W / np.max(np.abs(np.linalg.eigvals(W)))
    Win = rng.normal(0, 1, (n_res, U.shape[2])) * 0.5
    leaks = np.exp(rng.uniform(np.log(0.02), np.log(0.9), n_res))[None, :]
    h = np.zeros((len(U), n_res)); out = []
    for ti in range(U.shape[1]):
        h = (1 - leaks) * h + leaks * np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
        if ti % SUB == 0: out.append(h.copy())
    return np.stack(out, 1).reshape(len(U), -1)


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
for tag in ["GA", "Random", "BP"]:
    p = os.path.join(T1, "seed0", "%s_params.npy" % tag)
    if os.path.exists(p):
        PARAMS[tag] = np.load(p).astype(np.float32)
print("loaded type1 params:", {k: v.shape for k, v in PARAMS.items()})

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
NAMES = ["winriem", "T1_GA", "T1_Random", "T1_BP", "esn32",
         "win+T1_GA", "win+T1_Random", "win+esn32"]
RES = {t: {k: [] for k in NAMES} for t in ["place", "ph11"]}
print("\nType I substrate as EEG reservoir | GA(evolved) vs Random(control)\n")
for f in FILES:
    Xall, ph = load(f)
    tasks = {"place": (Xall[np.isin(ph, STOPS)],
                       np.where(np.isin(ph[np.isin(ph, STOPS)], ["d", "t"]), 1, 0), 2),
             "ph11": (Xall[np.isin(ph, PHON)],
                      np.array([PHON.index(v) for v in ph[np.isin(ph, PHON)]]), 6)}
    for tn, (X, y, K) in tasks.items():
        pc = PCA(8).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
        U = np.stack([pc.transform(z.T) for z in X], 0).astype(np.float32)
        U = (U - U.mean((0, 1))) / (U.std((0, 1)) + 1e-9)
        F = {"esn32": esn_features(U)}
        for tag in PARAMS:
            F["T1_" + tag] = type1_features(U, PARAMS[tag])
        acc = {k: [] for k in NAMES}
        for s in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                Rtr, Rte = winriem(X[tr], X[te], K)
                acc["winriem"].append(ev(Rtr, y[tr], Rte, y[te]))
                for nm in ["T1_GA", "T1_Random", "T1_BP", "esn32"]:
                    if nm in F or nm.replace("T1_", "") in PARAMS:
                        Z = F.get(nm)
                        if Z is None: continue
                        acc[nm].append(ev(Z[tr], y[tr], Z[te], y[te]))
                for nm, key in [("win+T1_GA", "T1_GA"), ("win+T1_Random", "T1_Random"),
                                ("win+esn32", "esn32")]:
                    if key not in F: continue
                    a, b = cat([Rtr, F[key][tr]], [Rte, F[key][te]])
                    acc[nm].append(ev(a, y[tr], b, y[te]))
        for k in NAMES:
            RES[tn][k].append(float(np.mean(acc[k])) if acc[k] else np.nan)
    print("  %-8s ph11 %s" % (os.path.basename(f).split("_")[0],
          " ".join("%.3f" % RES["ph11"][k][-1] for k in NAMES)), flush=True)

for tn, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tn, ch))
    base = np.array(RES[tn]["winriem"])
    for k in NAMES:
        a = np.array([v for v in RES[tn][k] if not np.isnan(v)])
        if len(a) == 0: continue
        _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base[:len(a)])
        print("  %-15s %.3f  vs-chance p=%.5f  vs-winriem %+.3f (p=%.4f) %s" % (
            k, a.mean(), p0, a.mean()-base.mean(), pb,
            "**BEATS**" if pb < 0.05 and a.mean() > base.mean() else ""))
    g, r = np.array(RES[tn]["T1_GA"]), np.array(RES[tn]["T1_Random"])
    if not np.isnan(g).all():
        _, pg = ttest_rel(g, r)
        print("  --- GA vs Random substrate: %+.3f (p=%.4f) %s" % (
            g.mean()-r.mean(), pg, "EVOLUTION HELPS" if pg < 0.05 and g.mean() > r.mean() else "no effect"))
