"""
Type I evolution where survival REQUIRES phoneme-relevant inference.

The scalar task did not transfer because survival only needed p(t)'s short-term value, which
carries no phoneme identity (ceiling test: 1-D observable -> 11-phoneme 0.101 vs chance 0.091).
Here the environment is built so surviving and discriminating are the same problem, following
embodied.py's four conditions:

    (a) genuine q-dependence      regulation is gated by where the agent is
    (b) |h| exceeds a_r's reach   one localized a_r cannot cover K channels
    (c) spatial structure slow    channel positions are static, p is fast
    (d) q is NOT observed         the agent must path-integrate its own motor history

    e_k-dot = -gamma e_k + p_k(t) + kappa * G(q - k) * tanh(a_r(t - tau))
    q-dot   = v_scale * a_e(t - tau)                      body, unobserved
    xi      = Izhikevich, sees only e (K channels via the K-window wiring)
    a_r     = tanh(w_out . rate) * 2      regulation magnitude
    a_e     = tanh(w_e   . rate) * 2      where to apply it
    death when max_k |e_k| > EMAX

Control: --embodied 0 makes the gate uniform. Same wiring and parameter count, no localization.
Genome = n_params(32) + 32, matching embodied.n_params_embodied.
"""
import argparse
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, vmap, jit

from search import run_cmaes, run_selection_only

N, NT, TAU, KAP, DEC, CLIP, EMAX = 32, 10, 8, 1.0, 0.98, 10.0, 4.0
DT_IZH, V_THRESH, I_TONIC, INPUT_SCALE, SYN_DECAY = 0.5, 30.0, 2.95, 30.0, 0.8
P_STD = 0.3
V_SCALE, Q_BOUND, SHELTER_W = 0.30, 6.0, 1.2
ENV_TAU = 150.0


def n_params(n=N):
    return n * n + 7 * n


def n_params_embodied(n=N):
    return n_params(n) + n


def split_genome(flat, n=N):
    i = 0
    W = flat[i:i+n*n].reshape(n, n); i += n*n
    b = flat[i:i+n]; i += n
    ns = n // 4
    w_in = flat[i:i+n] * jnp.zeros(n).at[:ns].set(1.0); i += n
    w_out = flat[i:i+n] * jnp.zeros(n).at[-ns:].set(1.0); i += n
    a = flat[i:i+n]; i += n
    bp = flat[i:i+n]; i += n
    c = flat[i:i+n]; i += n
    d = flat[i:i+n]; i += n
    w_e = flat[i:i+n]
    mask = ((jnp.arange(n) >= n - 2*ns) & (jnp.arange(n) < n - ns)).astype(flat.dtype)
    return W, b, w_in, w_out, a, bp, c, d, w_e * mask


def expand_obs(e_vec, n=N):
    ns = n // 4
    k = e_vec.shape[0]
    per = ns // k
    parts = [jnp.full(per, e_vec[j]) for j in range(k)]
    rem = ns - k * per
    if rem > 0:
        parts.append(jnp.full(rem, e_vec[0]))
    return jnp.concatenate([jnp.concatenate(parts), jnp.zeros(n - ns)])


def survive(flat, p, embodied=True):
    W, b, w_in, w_out, a, bp, c, d, w_e = split_genome(flat)
    k = p.shape[1]
    centres = jnp.arange(k, dtype=jnp.float32) - (k - 1) / 2.0

    def step(cr, pt):
        v, u, sp, syn, buf_r, buf_e, e, q, alive, steps = cr
        I_ext = w_in * expand_obs(e) * INPUT_SCALE

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
        a_r = jnp.tanh(jnp.dot(w_out, rate)) * 2.0
        a_e = jnp.tanh(jnp.dot(w_e, rate)) * 2.0
        land_r, land_e = buf_r[0], buf_e[0]
        buf_r = jnp.roll(buf_r, -1).at[-1].set(a_r)
        buf_e = jnp.roll(buf_e, -1).at[-1].set(a_e)
        q2 = jnp.clip(q + V_SCALE * land_e, -Q_BOUND, Q_BOUND)
        gate = jnp.where(embodied,
                         jnp.exp(-0.5 * ((q2 - centres) / SHELTER_W) ** 2),
                         jnp.ones(k) / jnp.sqrt(k))
        e2 = jnp.clip(DEC * e + pt + KAP * gate * jnp.tanh(land_r), -CLIP, CLIP)
        ok = alive & (jnp.max(jnp.abs(e2)) <= EMAX)
        e2 = jnp.where(ok, e2, e)
        return (v, u, sp, syn, buf_r, buf_e, e2, q2, ok, steps + ok), None

    init = (jnp.full(N, -65.0), jnp.zeros(N), jnp.zeros(N), jnp.zeros(N),
            jnp.zeros(TAU), jnp.zeros(TAU), jnp.zeros(k), 0.0, True, 0)
    (*_, steps), _ = lax.scan(step, init, p)
    return steps


def make_fitness(embodied):
    ms = jit(lambda f, P: jnp.mean(vmap(lambda p: survive(f, p, embodied))(P)))
    return ms, jit(lambda POP, P: vmap(lambda f: ms(f, P))(POP))


def hotspot(stream, tau=ENV_TAU, sharp=3.0):
    """Concentrate the perturbation on a slowly drifting channel (condition c).

    The slow envelope of each geometric channel sets where the danger is; the fast fluctuation
    rides on top. The hot spot then moves on a timescale the agent can follow."""
    n, k = stream.shape
    env = np.zeros((n, k))
    acc = np.zeros(k)
    al = np.exp(-1.0 / tau)
    for i in range(n):
        acc = al * acc + (1 - al) * np.abs(stream[i])
        env[i] = acc
    w = np.exp(sharp * (env - env.max(1, keepdims=True)))
    w = w / (w.sum(1, keepdims=True) + 1e-9)
    out = stream * w * k
    return out / (np.sqrt(np.mean(out.std(0) ** 2)) + 1e-9) * P_STD


def calibrate(stream, length, target_frac=0.35, seed=0):
    """Pick the perturbation amplitude so a do-nothing agent survives target_frac of an episode.

    The geometric channels are smooth, so any persistent component is amplified by
    1/(1-DEC) = 50; without calibration even a silent agent dies in ~10 steps and the task is
    unlearnable. Calibrating on the SILENT agent keeps the environment label-free."""
    z = jnp.zeros(n_params_embodied())
    lo, hi = 1e-4, 2.0
    for _ in range(18):
        mid = np.sqrt(lo * hi)
        P = episodes(stream / (stream.std() + 1e-9) * mid, 12, length, seed)
        surv = float(np.mean([survive(z, pp) for pp in P]))
        if surv < target_frac * length:
            hi = mid
        else:
            lo = mid
    return np.sqrt(lo * hi)


def episodes(stream, n_ep, length, seed):
    rng = np.random.default_rng(seed)
    st = rng.integers(0, len(stream) - length - 1, size=n_ep)
    return jnp.asarray(np.stack([stream[s:s+length] for s in st]), dtype=jnp.float32)


def main(args):
    stream = np.load(args.pstream)
    if stream.ndim == 1:
        stream = stream[:, None]
    stream = stream[:, :args.kobs]
    stream = stream / (np.sqrt(np.mean(stream.std(0) ** 2)) + 1e-9) * P_STD
    if args.hotspot:
        stream = hotspot(stream)
    if args.autoscale:
        amp = calibrate(stream, args.length, args.target_frac, args.seed)
        stream = stream / (stream.std() + 1e-9) * amp
        print("  autoscaled perturbation amplitude -> %.4f" % amp, flush=True)
    print("stream %s  per-channel std %.3f  embodied=%d  search=%s" % (
        stream.shape, stream.std(0).mean(), args.embodied, args.search), flush=True)

    Ptr = episodes(stream, args.eps, args.length, args.seed)
    Pte = episodes(stream, 64, args.length, args.seed + 10_000)
    ms, pop_f = make_fitness(bool(args.embodied))

    def fitness(X):
        return np.asarray(pop_f(jnp.asarray(X, dtype=jnp.float32), Ptr), dtype=float)

    hist, t0 = [], time.time()

    def cb(g, es, X, f, bx, bf):
        if g % args.log_every == 0 or g == args.gens:
            te = float(ms(jnp.asarray(bx, dtype=jnp.float32), Pte))
            sig = float(getattr(es, "sigma", np.nan))
            hist.append({"gen": int(g), "train": float(bf), "test": te, "sigma": sig})
            print("  gen %4d  train %7.2f  test %7.2f  sigma %.4f  [%.1fs]" % (
                g, float(bf), te, sig, time.time() - t0), flush=True)

    D = n_params_embodied()
    if args.search == "cmaes":
        best, bf, _ = run_cmaes(fitness, D, args.gens, pop=args.pop,
                                sigma0=args.sigma0, seed=args.seed, callback=cb)
    else:
        best, bf, _ = run_selection_only(fitness, D, args.gens, pop=args.pop,
                                         sigma=args.sigma, seed=args.seed, callback=cb)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, np.asarray(best))
    with open(args.out.replace(".npy", "_log.json"), "w") as fh:
        json.dump({"args": vars(args), "history": hist}, fh, indent=2)
    print("\nsaved %s   (max survival = %d)" % (args.out, args.length))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pstream", default="gstream_s1.npy")
    ap.add_argument("--kobs", type=int, default=4)
    ap.add_argument("--embodied", type=int, default=1)
    ap.add_argument("--hotspot", type=int, default=1)
    ap.add_argument("--autoscale", type=int, default=1)
    ap.add_argument("--target_frac", type=float, default=0.35)
    ap.add_argument("--search", choices=["cmaes", "selection"], default="cmaes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=800)
    ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--sigma0", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--eps", type=int, default=32)
    ap.add_argument("--length", type=int, default=800)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--out", default="result_eeg/emb_s0.npy")
    main(ap.parse_args())
