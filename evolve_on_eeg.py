"""
Re-evolve a Type I substrate on EEG-derived perturbations.

Framework unchanged (core/plant.py semantics):
    e-dot = -gamma e + p(t) + kappa tanh(a_r(t - tau)),  gamma = 1 - ENV_DECAY
    xi    = Izhikevich net, sees only e
    a_r   = tanh(w_out . rate) * 2
    fitness = survival steps while |e| <= EMAX

Only p(t) changes: instead of gen_perturbs (smooth Lorenz), p is derived from EEG.
This asks whether the Type I principle holds when the world is EEG statistics.

Run on a server; GA with pop=256 x 400 generations is the expensive part.
    python evolve_on_eeg.py --seed 0 --gens 400 --pop 256 --out results_eeg/seed0.npy
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

N, NT, TAU, KAP, DEC, CLIP, EMAX = 32, 10, 8, 1.0, 0.98, 10.0, 4.0
DT_IZH, V_THRESH, I_TONIC, INPUT_SCALE, SYN_DECAY = 0.5, 30.0, 2.95, 30.0, 0.8
P_STD = 0.3


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


def roll_survive(flat: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    W, b, w_in, w_out, a, bp, c, d = unpack(flat)
    ns = N // 4

    def step(cr, pt):
        v, u, sp, syn, buf, e, alive, steps = cr
        obs = jnp.concatenate([jnp.full(ns, e), jnp.zeros(N - ns)])
        I_ext = w_in * obs * INPUT_SCALE

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
        e2 = jnp.clip(DEC * e + pt + KAP * jnp.tanh(land), -CLIP, CLIP)
        ok = alive & (jnp.abs(e2) <= EMAX)
        e2 = jnp.where(ok, e2, e)
        return (v, u, sp, syn, buf, e2, ok, steps + ok), None

    init = (jnp.full(N, -65.0), jnp.zeros(N), jnp.zeros(N), jnp.zeros(N),
            jnp.zeros(TAU), 0.0, True, 0)
    (*_, steps), _ = lax.scan(step, init, p)
    return steps


mean_survival = jit(lambda flat, P: jnp.mean(vmap(lambda p: roll_survive(flat, p))(P)))
pop_survival = jit(lambda POP, P: vmap(lambda f: mean_survival(f, P))(POP))


def eeg_perturbations(data_dir: str, n_ep: int, length: int, seed: int) -> jnp.ndarray:
    files = sorted(glob.glob(os.path.join(data_dir, "sub-S*_ses-02.npz")))
    files = [f for f in files if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
    rng = np.random.default_rng(seed)
    segs = []
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
        pc1 = np.einsum("c,nct->nt", Vt[0], Xk)
        segs.append(pc1.reshape(-1))
    stream = np.concatenate(segs)
    stream = stream / (stream.std() + 1e-9) * P_STD
    starts = rng.integers(0, len(stream) - length - 1, size=n_ep)
    return jnp.asarray(np.stack([stream[s:s + length] for s in starts]), dtype=jnp.float32)


def tournament(key, fit, n_sel, k, pop):
    idx = jr.randint(key, (n_sel, k), 0, pop)
    return jnp.argmax(fit[idx], axis=1), idx


def evolve(args) -> None:
    key = jr.PRNGKey(args.seed)
    k1, k2, key = jr.split(key, 3)
    Ptrain = eeg_perturbations(args.data, args.eps, args.length, args.seed)
    Ptest = eeg_perturbations(args.data, 64, args.length, args.seed + 10_000)
    print("perturbations: train %s  test %s  std %.3f" % (
        Ptrain.shape, Ptest.shape, float(jnp.std(Ptrain))), flush=True)

    P = n_params()
    POP = jr.normal(k1, (args.pop, P)) * args.init_std
    best_hist = []
    t0 = time.time()
    for g in range(args.gens):
        fit = pop_survival(POP, Ptrain)
        order = jnp.argsort(fit)[::-1]
        elite = POP[order[:args.elite]]
        key, ks, km = jr.split(key, 3)
        sel_j, idx = tournament(ks, fit, args.pop - args.elite, args.k, args.pop)
        parents = POP[idx[jnp.arange(idx.shape[0]), sel_j]]
        noise = jr.normal(km, parents.shape) * args.sigma
        POP = jnp.concatenate([elite, parents + noise], axis=0)
        if g % args.log_every == 0 or g == args.gens - 1:
            te = float(mean_survival(elite[0], Ptest))
            best_hist.append({"gen": int(g), "train": float(fit[order[0]]), "test": te})
            print("  gen %4d  train %7.2f  test %7.2f   [%.1fs]" % (
                g, float(fit[order[0]]), te, time.time() - t0), flush=True)

    best = np.asarray(POP[0])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, best)
    with open(args.out.replace(".npy", "_log.json"), "w") as fh:
        json.dump({"args": vars(args), "history": best_hist}, fh, indent=2)
    print("\nsaved %s  (max possible survival = %d)" % (args.out, args.length))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=400)
    ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--elite", type=int, default=16)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--init_std", type=float, default=1.0)
    ap.add_argument("--eps", type=int, default=32)
    ap.add_argument("--length", type=int, default=800)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--out", default="results_eeg/seed0.npy")
    evolve(ap.parse_args())
