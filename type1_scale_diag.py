import warnings
warnings.filterwarnings("ignore")
import numpy as np
import jax.numpy as jnp
from jax import vmap

exec(open("type1_closedloop.py").read().split("PARAMS = {}")[0].replace(
    "FILES = [f for f", "_x=[f for f"))

GA = np.load("/Users/siwonyun/Work/programming/type1_learning/results/seeds/seed0.npy").astype(np.float32)
W, b, w_in, w_out, a, bp, c, d = unpack(jnp.asarray(GA))

X, ph = load("npz/sub-S04_ses-02.npz")
pc_raw = scalar_obs(X)[:, :, 0] / P_STD

print("=== baseline autonomous behaviour (p = 0) ===")
zero = jnp.zeros_like(jnp.asarray(pc_raw[:8]))
r0, e0 = vmap(lambda p: closed_loop_rates(p, W, b, w_in, w_out, a, bp, c, d))(zero)
e0 = np.asarray(e0)
print("  p=0 : e final %.3f  e std %.3f  range [%.2f, %.2f]" % (
    e0[:, -1].mean(), e0.std(), e0.min(), e0.max()))
print("  -> if e drifts to CLIP with zero input, the bias is the network's own a_r offset\n")

print("=== scale scan (target: |e| well inside EMAX=4, high across-trial rate variance) ===")
print("%8s %9s %9s %9s %9s %10s %10s" % ("p_std", "e_std", "e_min", "e_max", "sat_frac", "rate_var", "silent"))
best = None
for s in [0.02, 0.05, 0.1, 0.2, 0.3, 0.6]:
    P = jnp.asarray(pc_raw[:40] * s)
    r, e = vmap(lambda p: closed_loop_rates(p, W, b, w_in, w_out, a, bp, c, d))(P)
    r, e = np.asarray(r), np.asarray(e)
    sat = float(np.mean(np.abs(e) > 9.5))
    rv = float(r.mean(1).var(0).mean())
    sil = float(np.mean(r.mean((0, 1)) < 1e-6))
    print("%8.2f %9.3f %9.2f %9.2f %9.4f %10.6f %10.2f" % (
        s, e.std(), e.min(), e.max(), sat, rv, sil))
    if sat < 0.05 and (best is None or rv > best[1]):
        best = (s, rv)

print("\n=== KAP sweep at p_std=0.3 (reduce feedback authority) ===")
print("%8s %9s %9s %9s %10s" % ("KAP", "e_std", "e_min", "e_max", "rate_var"))


def loop_kap(p_seq, kap):
    ns = N // 4
    v0 = jnp.full(N, -65.0); u0 = jnp.zeros(N)
    sp0 = jnp.zeros(N); sy0 = jnp.zeros(N)

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
            v_n = vv + dv * DT_IZH
            u_n = uu + du * DT_IZH
            fired = v_n >= V_THRESH
            v_n = jnp.where(fired, c, v_n)
            u_n = jnp.where(fired, u_n + d, u_n)
            return (jnp.clip(v_n, -100.0, V_THRESH), u_n, fired.astype(vv.dtype), sy_n), fired.astype(vv.dtype)

        (v, u, sp, syn), all_sp = jax.lax.scan(tick, (v, u, sp, syn), None, length=NT)
        rate = jnp.mean(all_sp, axis=0)
        act = jnp.tanh(jnp.dot(w_out, rate)) * 2.0
        land = buf[0]
        buf = jnp.roll(buf, -1).at[-1].set(act)
        e2 = jnp.clip(ENV_DECAY * e + pt + kap * jnp.tanh(land), -CLIP, CLIP)
        return (v, u, sp, syn, buf, e2), (rate, e2)

    init = (v0, u0, sp0, sy0, jnp.zeros(TAU), 0.0)
    _, out = jax.lax.scan(step, init, p_seq)
    return out


import jax
for kap in [1.0, 0.5, 0.2, 0.05, 0.0]:
    P = jnp.asarray(pc_raw[:40] * 0.3)
    r, e = vmap(lambda p: loop_kap(p, kap))(P)
    r, e = np.asarray(r), np.asarray(e)
    print("%8.2f %9.3f %9.2f %9.2f %10.6f" % (
        kap, e.std(), e.min(), e.max(), r.mean(1).var(0).mean()))
