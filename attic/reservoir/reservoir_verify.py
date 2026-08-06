import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
N_RES, D_IN, SUB, LEAK, RHO = 200, 20, 20, 0.3, 1.0

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w]
    return Xb, df[m].reset_index(drop=True)

def reservoir_w(seed: int):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.1)
    W = W / np.max(np.abs(np.linalg.eigvals(W))) * RHO
    return W, rng.normal(0, 1, (N_RES, D_IN)) * 0.5

def run(U, W, Win, mode):
    n, T, _ = U.shape
    h = np.zeros((n, N_RES)); out = []
    for ti in range(T):
        inp = U[:, ti, :] @ Win.T
        if mode == "full":
            pre = np.tanh(h @ W.T + inp); h = (1-LEAK)*h + LEAK*pre
        elif mode == "linear":
            pre = h @ W.T + inp; h = (1-LEAK)*h + LEAK*pre
        elif mode == "no_recur":
            h = (1-LEAK)*h + LEAK*np.tanh(inp)
        elif mode == "no_mem":
            h = np.tanh(inp)
        elif mode == "mem_only":
            h = (1-LEAK)*h + LEAK*inp
        if ti % SUB == 0:
            out.append(h.copy())
    return np.stack(out, 1).reshape(n, -1)

est = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
def ms(F, y, seeds=3):
    return float(np.mean([cross_val_score(est, F, y, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                                          scoring="balanced_accuracy").mean() for s in range(seeds)]))

CFG = [("volt", None, None)] + [("seed%d" % s, "full", s) for s in range(5)] + \
      [("linear", "linear", 0), ("no_recur", "no_recur", 0), ("no_mem", "no_mem", 0), ("mem_only", "mem_only", 0)]
R = {c[0]: [] for c in CFG}
Ws = {s: reservoir_w(s) for s in range(5)}
print("Reservoir verification | place-stops | seeds + ablation ladder\n")
for f in FILES:
    Xb, ds = load(f)
    ph = ds.phoneme.values
    mask = np.isin(ph, ["b", "p", "d", "t"])
    y = np.where(np.isin(ph[mask], ["d", "t"]), 1, 0)
    p = PCA(D_IN).fit(Xb.transpose(0, 2, 1).reshape(-1, Xb.shape[1]))
    U = np.stack([p.transform(x.T) for x in Xb], 0)
    U = (U - U.mean((0, 1))) / (U.std((0, 1)) + 1e-9)
    for name, mode, seed in CFG:
        if mode is None:
            F = Xb[:, :, ::10].reshape(len(Xb), -1)
        else:
            F = run(U, *Ws[seed], mode)
        R[name].append(ms(F[mask], y))
    print("  %s  volt %.3f  seed0 %.3f  no_mem %.3f" % (
        os.path.basename(f).split("_")[0], R["volt"][-1], R["seed0"][-1], R["no_mem"][-1]), flush=True)

vo = np.array(R["volt"])
print("\n=== SEED REPRODUCIBILITY (full reservoir, rho=1.0) ===")
for s in range(5):
    a = np.array(R["seed%d" % s]); _, p = ttest_rel(a, vo)
    print("  seed%d  %.3f  vs-volt %+.3f  p=%.4f %s" % (s, a.mean(), a.mean()-vo.mean(), p, "*" if p < 0.05 else ""))
sm = np.mean([R["seed%d" % s] for s in range(5)], 0)
_, p = ttest_rel(sm, vo)
print("  MEAN over 5 seeds: %.3f  vs-volt %+.3f  p=%.4f" % (sm.mean(), sm.mean()-vo.mean(), p))

print("\n=== ABLATION (what drives the gain?) ===")
print("  voltage baseline           %.3f" % vo.mean())
for name, lbl in [("seed0", "full (recur+tanh+mem)"), ("linear", "linear (no tanh)"),
                  ("no_recur", "no recurrence"), ("no_mem", "no memory (tanh only)"), ("mem_only", "memory only (linear)")]:
    a = np.array(R[name]); _, p = ttest_rel(a, vo)
    print("  %-26s %.3f  vs-volt %+.3f  p=%.4f %s" % (lbl, a.mean(), a.mean()-vo.mean(), p, "*" if p < 0.05 else ""))
