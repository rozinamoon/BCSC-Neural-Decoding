import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
D_IN, N_RES, LEAK, SUB = 20, 200, 0.3, 20
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df.phoneme.values[m]

def make_w(seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.1)
    W = W / np.max(np.abs(np.linalg.eigvals(W)))
    return W, rng.normal(0, 1, (N_RES, D_IN)) * 0.5

def run_neuron(U, W, Win, kind):
    n, T, _ = U.shape
    h = np.zeros((n, N_RES)); v = np.full((n, N_RES), -65.0); u_ = np.full((n, N_RES), -13.0)
    trace = np.zeros((n, N_RES)); out = []
    for ti in range(T):
        I = h @ W.T + U[:, ti, :] @ Win.T if kind not in ("lif", "izh") else (trace @ W.T + U[:, ti, :] @ Win.T) * 12.0
        if kind == "tanh":
            h = (1-LEAK)*h + LEAK*np.tanh(I); st = h
        elif kind == "relu":
            h = (1-LEAK)*h + LEAK*np.maximum(I, 0); h = np.clip(h, 0, 10); st = h
        elif kind == "sine":
            h = (1-LEAK)*h + LEAK*np.sin(I); st = h
        elif kind == "quad":
            h = (1-LEAK)*h + LEAK*np.clip(I - I**3/3.0, -3, 3); st = h
        elif kind == "lif":
            v = 0.9*v + I
            sp = (v > 5.0).astype(float); v = np.where(sp > 0, 0.0, v)
            trace = 0.8*trace + sp; st = trace
        elif kind == "izh":
            for _ in range(2):
                v = v + 0.5*(0.04*v*v + 5*v + 140 - u_ + I)
                v = np.clip(v, -100, 40)
            u_ = u_ + 0.02*(0.2*v - u_)
            sp = (v >= 30).astype(float)
            v = np.where(sp > 0, -65.0, v); u_ = np.where(sp > 0, u_ + 8.0, u_)
            trace = 0.8*trace + sp; st = trace
        if ti % SUB == 0:
            out.append(st.copy())
    return np.stack(out, 1).reshape(n, -1)

def feats(Xtr, Xte, W, Win, kind):
    p = PCA(D_IN).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    def prep(Z):
        U = np.stack([p.transform(z.T) for z in Z], 0)
        return (U - U.mean((0, 1))) / (U.std((0, 1)) + 1e-9)
    return run_neuron(prep(Xtr), W, Win, kind), run_neuron(prep(Xte), W, Win, kind)

def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5))

KINDS = ["tanh", "relu", "sine", "quad", "lif", "izh"]
RES = {t: {k: [] for k in KINDS} for t in ["place", "ph11"]}
W, Win = make_w(0)
print("NEURON TYPE comparison in reservoir | place (chance .5) & 11-phoneme (chance .091)\n")
for f in FILES:
    Xall, ph = load(f)
    k1 = np.isin(ph, ["b", "p", "d", "t"])
    k2 = np.isin(ph, PHON)
    tasks = {"place": (Xall[k1], np.where(np.isin(ph[k1], ["d", "t"]), 1, 0)),
             "ph11": (Xall[k2], np.array([PHON.index(v) for v in ph[k2]]))}
    for tname, (X, y) in tasks.items():
        for kind in KINDS:
            a = []
            for s in range(2):
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                    Ftr, Fte = feats(X[tr], X[te], W, Win, kind)
                    c = lr(); c.fit(Ftr, y[tr])
                    a.append(balanced_accuracy_score(y[te], c.predict(Fte)))
            RES[tname][kind].append(float(np.mean(a)))
    print("  %-8s place %s | ph11 %s" % (os.path.basename(f).split("_")[0],
          " ".join("%s:%.3f" % (k[:4], RES["place"][k][-1]) for k in KINDS),
          " ".join("%s:%.3f" % (k[:4], RES["ph11"][k][-1]) for k in KINDS)), flush=True)

for tname, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tname, ch))
    base = np.array(RES[tname]["tanh"])
    for k in KINDS:
        a = np.array(RES[tname][k]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-5s %.3f  vs-chance p=%.5f  vs-tanh %+.3f (p=%.4f)" % (k, a.mean(), p0, a.mean()-base.mean(), pb))
