import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
SIZES = [20, 40, 60, 90, 120, 150]

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "category", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w]
    return Xb, df[m].reset_index(drop=True)

def dda(x, tau1=4, tau2=20):
    n, c, T = x.shape; m = T - tau2 - 1
    ud = x[:, :, tau2+1:tau2+1+m] - x[:, :, tau2:tau2+m]
    a = x[:, :, tau2+1-tau1:tau2+1-tau1+m]; b = x[:, :, 1:1+m]
    D = np.stack([a, b, a*a], -1)
    DtD = np.einsum("ncmi,ncmj->ncij", D, D) + 1e-8*np.eye(3)
    Dty = np.einsum("ncmi,ncm->nci", D, ud)
    coef = np.einsum("bij,bj->bi", np.linalg.inv(DtD.reshape(-1, 3, 3)), Dty.reshape(-1, 3)).reshape(n, c, 3)
    return coef.reshape(n, -1)

def reps(Xb):
    hi = Xb[:, :, ::10].reshape(len(Xb), -1)
    return {"voltage_hi(%dD)" % hi.shape[1]: hi,
            "voltage_PCA20": None,
            "DDA(183D)": dda(Xb)}

def curve(F, y, sizes, n_rep=12, pca=None):
    out = {}
    for ns in sizes:
        if ns >= len(y) - 20:
            continue
        accs = []
        sss = StratifiedShuffleSplit(n_splits=n_rep, train_size=ns, test_size=40, random_state=0)
        for tr, te in sss.split(F, y):
            Xtr, Xte = F[tr], F[te]
            if pca is not None:
                p = PCA(min(pca, len(tr)-2)).fit(Xtr)
                Xtr, Xte = p.transform(Xtr), p.transform(Xte)
            clf = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
            clf.fit(Xtr, y[tr])
            accs.append(balanced_accuracy_score(y[te], clf.predict(Xte)))
        out[ns] = float(np.mean(accs))
    return out

ALL = {}
for f in FILES:
    Xb, ds = load(f)
    y = np.where(ds.category.values != "vowels", 1, 0)
    hi = Xb[:, :, ::10].reshape(len(Xb), -1)
    dd = dda(Xb)
    for name, F, pc in [("voltage_hi", hi, None), ("voltage_PCA20", hi, 20), ("DDA", dd, None)]:
        c = curve(F, y, SIZES, pca=pc)
        ALL.setdefault(name, []).append(c)
    print("  %s done" % os.path.basename(f).split("_")[0], flush=True)

print("\n=== LEARNING CURVE (category, within-subject, test=40 held-out) ===")
print("  %-16s %s" % ("representation", "  ".join("n=%-5d" % s for s in SIZES)))
for name, lst in ALL.items():
    row = []
    for s in SIZES:
        v = [c[s] for c in lst if s in c]
        row.append("%.3f " % np.mean(v) if v else "  -   ")
    print("  %-16s %s" % (name, "  ".join(row)))

print("\n  interpretation: still rising at n=150 => ESTIMATION-limited (better representation helps)")
print("                  flat => INFORMATION-limited (ceiling is real)")
