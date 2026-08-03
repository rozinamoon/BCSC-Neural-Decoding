"""
Type I evolution with a geometric, multi-channel observation.

Framework unchanged in form:
    e_k-dot = -gamma e_k + p_k(t) + kappa tanh(a_r(t - tau)),   k = 1..K
    xi      = Izhikevich net, sees only e (K channels spread over the N//4 sensory neurons)
    a_r     = tanh(w_out . rate) * 2      (one shared regulation channel)
    death when max_k |e_k| > EMAX ;  fitness = survival steps

Two things change relative to evolve_on_eeg.py:
  * K > 1: the observation carries several channels, wired exactly as plant.py's K-window
  * p_k(t) is Riemannian: sliding-window covariance -> tangent space at the global Karcher
    mean -> top-K principal directions.  Unsupervised, so no label leakage.

Controls (mirroring plant.py's window='dup'):
    --obs dup   K copies of channel 0: same wiring, no extra information

    python evolve_geom.py --extract --data npz --out gstream.npy --kobs 8
    python evolve_geom.py --pstream gstream.npy --kobs 8 --gens 800 --pop 512
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap, jit

jax.config.update("jax_enable_x64", False)

from search import run_cmaes, run_selection_only

N, NT, TAU, KAP, DEC, CLIP, EMAX = 32, 10, 8, 1.0, 0.98, 10.0, 4.0
DT_IZH, V_THRESH, I_TONIC, INPUT_SCALE, SYN_DECAY = 0.5, 30.0, 2.95, 30.0, 0.8
P_STD = 0.3
UNROLL = 8
WIN, STRIDE, NPCA_COV = 50, 1, 10


def n_params(n: int = N) -> int:
    return n * n + 7 * n


def unpack(flat: jnp.ndarray, n: int = N):
    i = 0
    W = flat[i:i + n * n].reshape(n, n); i += n * n
    b = flat[i:i + n]; i += n
    ns = n // 4
    w_in = flat[i:i + n] * jnp.zeros(n).at[:ns].set(1.0); i += n
    w_out = flat[i:i + n] * jnp.zeros(n).at[-ns:].set(1.0); i += n
    a = flat[i:i + n]; i += n
    bp = flat[i:i + n]; i += n
    c = flat[i:i + n]; i += n
    d = flat[i:i + n]; i += n
    return W, b, w_in, w_out, a, bp, c, d


def expand_obs(e_vec: jnp.ndarray, n: int = N) -> jnp.ndarray:
    ns = n // 4
    k = e_vec.shape[0]
    per = ns // k
    parts = [jnp.full(per, e_vec[j]) for j in range(k)]
    rem = ns - k * per
    if rem > 0:
        parts.append(jnp.full(rem, e_vec[0]))
    return jnp.concatenate([jnp.concatenate(parts), jnp.zeros(n - ns)])


def roll_survive(flat: jnp.ndarray, p: jnp.ndarray, dup: bool = False) -> jnp.ndarray:
    W, b, w_in, w_out, a, bp, c, d = unpack(flat)
    k = p.shape[1]

    def step(cr, pt):
        v, u, sp, syn, buf, e, alive, steps = cr
        e_obs = jnp.full(k, e[0]) if dup else e
        I_ext = w_in * expand_obs(e_obs) * INPUT_SCALE

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

        (v, u, sp, syn), all_sp = lax.scan(tick, (v, u, sp, syn), None, length=NT, unroll=NT)
        rate = jnp.mean(all_sp, axis=0)
        act = jnp.tanh(jnp.dot(w_out, rate)) * 2.0
        land = buf[0]
        buf = jnp.roll(buf, -1).at[-1].set(act)
        e2 = jnp.clip(DEC * e + pt + KAP * jnp.tanh(land), -CLIP, CLIP)
        ok = alive & (jnp.sqrt(jnp.mean(e2 ** 2)) <= EMAX)
        e2 = jnp.where(ok, e2, e)
        return (v, u, sp, syn, buf, e2, ok, steps + ok), None

    init = (jnp.full(N, -65.0), jnp.zeros(N), jnp.zeros(N), jnp.zeros(N),
            jnp.zeros(TAU), jnp.zeros(k), True, 0)
    (*_, steps), _ = lax.scan(step, init, p, unroll=UNROLL)
    return steps


def make_fitness(dup: bool):
    ms = jit(lambda flat, P: jnp.mean(vmap(lambda p: roll_survive(flat, p, dup))(P)))
    return ms, jit(lambda POP, P: vmap(lambda f: ms(f, P))(POP))


def spd_log(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-10, None)
    return (V * np.log(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def spd_pow(C, a):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-10, None)
    return (V * (w ** a)[..., None, :]) @ np.swapaxes(V, -1, -2)


def geometric_stream(data_dir: str, kobs: int) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(data_dir, "sub-S*_ses-02.npz")))
    files = [f for f in files if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
    if not files:
        raise SystemExit("no npz in %s" % data_dir)
    chunks = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        X, t = d["X"].astype(float), d["times"]
        meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target"]})
        k = ((meta.task == "single-phoneme") & meta.tms_target.str.startswith("control")).values
        wm = (t >= 0.0) & (t <= 0.5)
        Xk = X[k][:, :, wm]
        flat = Xk.transpose(0, 2, 1).reshape(-1, Xk.shape[1])
        flat = flat - flat.mean(0)
        _, _, Vt = np.linalg.svd(flat, full_matrices=False)
        Z = np.einsum("kc,nct->nkt", Vt[:NPCA_COV], Xk)
        starts = range(0, Z.shape[2] - WIN + 1, STRIDE)
        Cs = np.stack([np.einsum("nkt,nlt->nkl", Z[:, :, s:s+WIN], Z[:, :, s:s+WIN]) / WIN
                       for s in starts], 1)
        dd = Cs.shape[-1]
        Cs = Cs + 1e-6 * np.trace(Cs, axis1=-2, axis2=-1)[..., None, None] / dd * np.eye(dd)
        M = Cs.reshape(-1, dd, dd).mean(0)
        for _ in range(10):
            Mh, Mih = spd_pow(M, 0.5), spd_pow(M, -0.5)
            S = spd_log(np.einsum("ij,njk,kl->nil", Mih, Cs.reshape(-1, dd, dd), Mih)).mean(0)
            w_, V_ = np.linalg.eigh(S)
            M = Mh @ (V_ * np.exp(w_)) @ V_.T @ Mh
        Mih = spd_pow(M, -0.5)
        L = spd_log(np.einsum("ij,njk,kl->nil", Mih, Cs.reshape(-1, dd, dd), Mih))
        iu = np.triu_indices(dd)
        scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
        T_ = (L[:, iu[0], iu[1]] * scale).reshape(Cs.shape[0], Cs.shape[1], -1)
        chunks.append(T_)
        print("  %-10s windows %d  tangent dim %d" % (
            os.path.basename(f).split("_")[0], T_.shape[1], T_.shape[2]), flush=True)
    A = np.concatenate(chunks, 0)
    flatA = A.reshape(-1, A.shape[2])
    flatA = flatA - flatA.mean(0)
    _, _, Vt = np.linalg.svd(flatA, full_matrices=False)
    proj = (A @ Vt[:kobs].T).reshape(-1, kobs)
    proj = proj / (proj.std(0, keepdims=True) + 1e-9) * (P_STD / np.sqrt(kobs))
    return proj.astype(np.float32)


def episodes(stream: np.ndarray, n_ep: int, length: int, seed: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    if len(stream) <= length + 1:
        raise SystemExit("stream too short (%d) for length %d" % (len(stream), length))
    st = rng.integers(0, len(stream) - length - 1, size=n_ep)
    return jnp.asarray(np.stack([stream[s:s + length] for s in st]), dtype=jnp.float32)


def evolve(args) -> None:
    stream = np.load(args.pstream)
    if stream.ndim == 1:
        stream = stream[:, None]
    stream = stream[:, :args.kobs]
    stream = stream / (np.sqrt(np.mean(stream.std(0) ** 2)) + 1e-9) * (P_STD / np.sqrt(stream.shape[1]))
    print("stream %s  std %.3f  kobs %d  obs-mode %s" % (
        stream.shape, stream.std(), stream.shape[1], args.obs), flush=True)

    Ptrain = episodes(stream, args.eps, args.length, args.seed)
    Ptest = episodes(stream, 64, args.length, args.seed + 10_000)
    mean_surv, pop_surv = make_fitness(args.obs == "dup")

    def fitness(X):
        return np.asarray(pop_surv(jnp.asarray(X, dtype=jnp.float32), Ptrain), dtype=float)

    hist, t0 = [], time.time()

    def cb(g, es, X, f, best_x, best_f):
        if g % args.log_every == 0 or g == args.gens:
            te = float(mean_surv(jnp.asarray(best_x, dtype=jnp.float32), Ptest))
            sig = getattr(es, "sigma", float("nan"))
            hist.append({"gen": int(g), "train": float(best_f), "test": te, "sigma": float(sig)})
            print("  gen %4d  train %7.2f  test %7.2f  sigma %.4f  [%.1fs]" % (
                g, float(best_f), te, sig, time.time() - t0), flush=True)

    print("search = %s" % args.search, flush=True)
    if args.search == "cmaes":
        best, bf, _ = run_cmaes(fitness, n_params(), args.gens, pop=args.pop,
                                sigma0=args.sigma0, seed=args.seed, callback=cb)
    else:
        best, bf, _ = run_selection_only(fitness, n_params(), args.gens, pop=args.pop,
                                         sigma=args.sigma, elite_frac=args.elite_frac,
                                         seed=args.seed, init_scale=args.init_std, callback=cb)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, np.asarray(best))
    with open(args.out.replace(".npy", "_log.json"), "w") as fh:
        json.dump({"args": vars(args), "history": hist}, fh, indent=2)
    print("\nsaved %s   (max survival = %d)" % (args.out, args.length))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--data", default="npz")
    ap.add_argument("--pstream", default="gstream.npy")
    ap.add_argument("--kobs", type=int, default=8)
    ap.add_argument("--obs", choices=["info", "dup"], default="info")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=800)
    ap.add_argument("--pop", type=int, default=512)
    ap.add_argument("--search", choices=["cmaes", "selection"], default="cmaes")
    ap.add_argument("--sigma0", type=float, default=1.0)
    ap.add_argument("--elite_frac", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--init_std", type=float, default=1.0)
    ap.add_argument("--eps", type=int, default=32)
    ap.add_argument("--length", type=int, default=800)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--out", default="result_eeg/geom_seed0.npy")
    a = ap.parse_args()
    if a.extract:
        s = geometric_stream(a.data, a.kobs)
        np.save(a.pstream, s)
        print("\nsaved %s  shape %s  %.1f MB" % (a.pstream, s.shape, s.nbytes / 1e6))
    else:
        evolve(a)
