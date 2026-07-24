import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
N_RES, D_IN, SUB = 200, 20, 20
RHOS = [0.5, 0.9, 1.0, 1.1, 1.3]

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "category", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w]
    return Xb, df[m].reset_index(drop=True)

def make_reservoir(rho: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.1)
    ev = np.max(np.abs(np.linalg.eigvals(W)))
    W = W / ev * rho
    Win = rng.normal(0, 1, (N_RES, D_IN)) * 0.5
    return W, Win

def run_reservoir(U: np.ndarray, W: np.ndarray, Win: np.ndarray, leak: float = 0.3) -> np.ndarray:
    n, T, d = U.shape
    h = np.zeros((n, N_RES))
    states = []
    for ti in range(T):
        pre = h @ W.T + U[:, ti, :] @ Win.T
        h = (1 - leak) * h + leak * np.tanh(pre)
        if ti % SUB == 0:
            states.append(h.copy())
    return np.stack(states, 1).reshape(n, -1)

volt = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
def ms(F, y):
    return float(np.mean([cross_val_score(volt, F, y, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                                          scoring="balanced_accuracy").mean() for s in range(3)]))

RES = {t: {"volt": [], **{r: [] for r in RHOS}} for t in ["cat", "pl"]}
resv = {r: make_reservoir(r) for r in RHOS}
print("Reservoir (fixed chaotic dynamics, trained readout only) | N=%d, rho sweep %s\n" % (N_RES, RHOS))
for f in FILES:
    Xb, ds = load(f)
    p = PCA(D_IN).fit(Xb.transpose(0, 2, 1).reshape(-1, Xb.shape[1]))
    U = np.stack([p.transform(x.T) for x in Xb], 0)
    U = (U - U.mean((0, 1))) / (U.std((0, 1)) + 1e-9)
    feats = {r: run_reservoir(U, *resv[r]) for r in RHOS}
    Fv = Xb[:, :, ::10].reshape(len(Xb), -1)
    ph = ds.phoneme.values
    for cond, mask, yv in [("cat", np.ones(len(ds), bool), np.where(ds.category.values != "vowels", 1, 0)),
                            ("pl", np.isin(ph, ["b", "p", "d", "t"]), np.where(np.isin(ph, ["d", "t"]), 1, 0))]:
        yy = yv[mask]
        RES[cond]["volt"].append(ms(Fv[mask], yy))
        for r in RHOS:
            RES[cond][r].append(ms(feats[r][mask], yy))
    print("  %s  cat volt %.3f | rho: %s" % (os.path.basename(f).split("_")[0], RES["cat"]["volt"][-1],
          " ".join("%.2f:%.3f" % (r, RES["cat"][r][-1]) for r in RHOS)), flush=True)

print("\n=== GROUP (within-subject) ===")
for cond, lbl in [("cat", "category"), ("pl", "place-stops")]:
    vo = np.array(RES[cond]["volt"])
    print("  %s   voltage %.3f" % (lbl, vo.mean()))
    for r in RHOS:
        a = np.array(RES[cond][r]); _, pc = ttest_rel(a, vo); _, p0 = ttest_1samp(a, 0.5)
        tag = "EDGE" if abs(r - 1.0) < 0.01 else ("chaotic" if r > 1 else "stable")
        print("    rho=%.1f (%-7s) %.3f  vs-chance p=%.3f  vs-volt %+.3f (p=%.3f) %s" % (
            r, tag, a.mean(), p0, a.mean() - vo.mean(), pc, "**BEATS VOLTAGE**" if pc < 0.05 and a.mean() > vo.mean() else ""))
