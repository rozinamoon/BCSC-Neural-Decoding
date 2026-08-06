
"""
One object per trial: space, dynamics, evoked deflection and time order in the same matrix,
and one operator that couples space to time instead of computing them apart.

Two things this repo built but never joined.

1. The melted SPD object (unified.py, unified2*.py, ~15 variants).

       Z(t) = [x(t); x(t-tau); ...; x(t-(p-1)tau); 1]
       C    = (1/T) sum_t Z Z^T

   It lost alone (ph11 0.224 vs 0.246 for separate-then-concatenate) and only won when
   concatenated back on (0.273). Its parameters d, p and the constant row were swept; tau was
   fixed at 20 throughout, the spatial basis was always PCA, and it was only ever run
   per-subject. The structural gap is visible in the construction: the constant row keeps the
   window *mean* and a covariance averages over t, so time order -- the one predictor this
   repo has confirmed nine times -- is not in the object.

   The constant row is the k = 0 member of a temporal basis. Carry the rest (one_KT):
   the cross blocks <mode_i, phi_k> are the projection of that mode's waveform onto temporal
   basis k, i.e. the time-resolved evoked response, inside the same matrix.

2. The reservoirs (reservoir*.py, 6 files). Every one draws W as a random sparse matrix and
   rescales its spectral radius; the only non-random variant, delay_geom_reservoir.geom_W, is
   built from class-discriminant eigenvectors of the data. None of them ever reads
   channel_positions_xyz, and none was ever combined with the SPD object.

   Generalised synchronisation cannot be what makes a reservoir work here: it needs a driver
   with an attractor and a washout long enough to forget initial conditions, while a trial is
   a one-shot impulse response 256 samples long whose informative part *is* the initial
   transient. The measured signature agrees -- neuron type and spectral radius were both
   null, which is what a reservoir looks like when it is low-pass filtering rather than
   synchronising.

   What is left is worth testing and never was: make W the scalp itself.

       r(t+1) = (1-a) r(t) + a f( W r(t) + x(t) ),   W = -L / rho(L),  W_in = I

   one node per electrode, no random input mixing. In the linear regime the eigenmodes of
   this system are exactly the Laplacian eigenmodes and mode k has pole (1-a) - a rho lam_k,
   so spatial frequency *is* the time constant: broad fields integrate slowly, focal patterns
   decay fast. Space and time are one operator rather than two feature blocks. The control
   that decides whether this is geometry or just recurrence is the same reservoir with W drawn
   at random at matched rho, leak and nonlinearity.

Arms, identical folds, each with the hier2 head (lambda noise sharing, gamma class sharing):
    one_K0        the unified object with the constant row only     == unified.py, control
    one_KT        the same object with the full temporal basis      == the fix
    blocks        the current separate-then-concatenate pipeline    == today's champion
    one_KT+blocks both
    res_lap       scalp-Laplacian reservoir -> unified object with temporal basis
    res_rand      matched random reservoir  -> same                 == geometry control
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.model_selection import StratifiedKFold

from sdcore import (DATA, basis_modes, head, laplacian_modes, lda_persubj, lda_pooled,
                    load_pooled, montage_W, pca_blocks, reservoir, tangent, unified_spd)

# ------------------------------------------------------------------ pieces

# ------------------------------------------------------------------ fold

def _fold(args):
    (X, V, pos, y, sid, tr, te, kind, d, lo, p, tau, K_time, phi, K_win,
     rho, leak, gain, nl, n_cls, n_sub, inner, seed, arms) = args
    U = basis_modes(kind, X, tr, V, d, lo)
    Z = np.einsum("ck,nct->nkt", U, X)

    reps = {}
    if "one_K0" in arms:
        reps["one_K0"] = tangent(unified_spd(Z, tr, p, tau, 0, phi), tr, sid)
    if arms & {"one_KT", "one_KT+blocks"}:
        reps["one_KT"] = tangent(unified_spd(Z, tr, p, tau, K_time, phi), tr, sid)
    if arms & {"blocks", "one_KT+blocks"}:
        reps["blocks"] = pca_blocks(X, tr, d, K_win, p, tau, sid=sid,
                                        subject_center=True)
    if "one_KT+blocks" in arms:
        reps["one_KT+blocks"] = np.hstack([reps["one_KT"], reps["blocks"]])
    for nm, wk in [("res_lap", "lap"), ("res_rand", "rand")]:
        if nm not in arms and not (nm == "res_lap" and
                                   arms & {"res_lap+one_KT", "res_lap+blocks"}):
            continue
        R = reservoir(X, montage_W(wk, pos, rho, seed=0), leak, gain, nl)
        Zr = np.einsum("ck,nct->nkt", U, R)
        reps[nm] = tangent(unified_spd(Zr, tr, p, tau, K_time, phi), tr, sid)
    if "res_lap+one_KT" in arms:
        reps["res_lap+one_KT"] = np.hstack([reps["res_lap"], reps["one_KT"]])
    if "res_lap+blocks" in arms:
        reps["res_lap+blocks"] = np.hstack([reps["res_lap"], reps["blocks"]])
    reps = {k: v for k, v in reps.items() if k in arms}

    out = {}
    for nm, F in reps.items():
        acc, lgk = head(F, y, sid, tr, te, n_cls, n_sub, inner, seed)
        out[nm] = acc
        out[nm + "_lgk"] = lgk
        out[nm + "_persubjLDA"] = lda_persubj(F, y, sid, tr, te, n_cls, 3 * n_cls)
        out[nm + "_pooledLDA"] = lda_pooled(F, y, tr, te)
    out["dim"] = {k: v.shape[1] for k, v in reps.items()}
    return out


ARMS = ["one_K0", "one_KT", "blocks", "one_KT+blocks", "res_lap", "res_rand",
        "res_lap+one_KT", "res_lap+blocks"]


def main(a):
    X, y, tok, sid, pos, n_sub = load_pooled(a.task, a.data, normalise=True)
    V, _ = laplacian_modes(pos)
    n_cls = len(np.unique(y)); chance = 1.0 / n_cls
    print("task=%s trials=%d subjects=%d classes=%d | basis=%s d=%d lo=%d p=%d tau=%d "
          "K_time=%d phi=%.2f | reservoir rho=%.2f leak=%.2f gain=%.2f nl=%s | side=%d" % (
              a.task, len(X), n_sub, n_cls, a.basis, a.d, a.lo, a.p, a.tau, a.K_time, a.phi,
              a.rho, a.leak, a.gain, a.nl, a.p * a.d + a.K_time + 1), flush=True)

    arms = [x for x in ARMS if x in a.arms.split(",")] if a.arms else ARMS
    strat = np.array(["%d_%d" % (s, c) for s, c in zip(sid, y)])
    jobs = []
    for sd in range(a.seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, strat):
            jobs.append((X, V, pos, y, sid, tr, te, a.basis, a.d, a.lo, a.p, a.tau,
                         a.K_time, a.phi, a.windows, a.rho, a.leak, a.gain, a.nl,
                         n_cls, n_sub, a.inner, sd, set(arms)))

    NAMES = arms + [x + s for x in arms for s in ("_persubjLDA", "_pooledLDA")]
    R = {k: [] for k in NAMES}
    LGK = {k: [] for k in arms}
    dims = None
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for out in ex.map(_fold, jobs):
            for k in NAMES:
                R[k].append(out[k])
            for k in arms:
                LGK[k].append(out[k + "_lgk"])
            dims = out["dim"]
            print("  fold  " + "  ".join("%s %.3f" % (k, out[k]) for k in arms), flush=True)

    print()
    print("  tangent dims: " + "  ".join("%s %d" % (k, dims[k]) for k in arms))
    print()
    print("=== %s (chance %.3f) ===" % (a.task, chance))
    print("%-16s %8s %8s %8s | %5s %5s %5s" % (
        "representation", "hier2", "persubj", "pooled", "lam*", "gam*", "kap*"))
    for k in arms:
        lg = np.mean(LGK[k], axis=0)
        print("%-16s %8.3f %8.3f %8.3f | %5.2f %5.2f %5.2f" % (
            k, np.mean(R[k]), np.mean(R[k + "_persubjLDA"]), np.mean(R[k + "_pooledLDA"]),
            lg[0], lg[1], lg[2]))
    print()
    ref = np.array(R[a.ref if a.ref in arms else arms[0]], float)
    for k in sorted(arms, key=lambda k: -np.mean(R[k])):
        v = np.array(R[k], float)
        _, p0 = ttest_1samp(v, chance)
        _, pb = ttest_rel(v, ref)
        print("  %-16s %.3f  vs-chance p=%.7f  vs-blocks %+.3f (p=%.4f) %s" % (
            k, v.mean(), p0, v.mean() - ref.mean(), pb,
            "**BEATS**" if pb < 0.05 and v.mean() > ref.mean() else ""))
    if {"res_lap", "res_rand"} <= set(arms):
        dif = np.array(R["res_lap"], float) - np.array(R["res_rand"], float)
        _, pd_ = ttest_1samp(dif, 0.0)
        print()
        print("  res_lap - res_rand: %+.4f  folds above zero %d/%d  paired p=%.4f" % (
            dif.mean(), int((dif > 0).sum()), len(dif), pd_))
    print()
    print("  one_K0 vs one_KT  -- value of putting time order inside the single object")
    print("  res_lap vs res_rand -- value of the montage in the recurrent operator")
    print("  res_lap vs one_KT -- value of coupling space to time through one operator")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ph11", choices=["ph11", "cv", "place"])
    ap.add_argument("--data", default=DATA)   # repo-root/npz2
    ap.add_argument("--basis", default="pca", choices=["pca", "lap"])
    ap.add_argument("--d", type=int, default=14)
    ap.add_argument("--lo", type=int, default=20)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--tau", type=int, default=20)
    ap.add_argument("--K_time", type=int, default=7)
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--leak", type=float, default=0.3)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--nl", default="tanh", choices=["tanh", "lin"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--arms", default="")
    ap.add_argument("--ref", default="blocks")
    ap.add_argument("--workers", type=int, default=5)
    main(ap.parse_args())
