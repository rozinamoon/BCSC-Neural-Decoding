import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.spatialfilters import Xdawn

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w]
    return Xb, df.phoneme.values[m]

def win_riem(Str, Ste, K=4, npca=None):
    if npca:
        p = PCA(npca).fit(Str.transpose(0, 2, 1).reshape(-1, Str.shape[1]))
        pr = lambda Z: np.stack([p.transform(z.T).T for z in Z], 0)
        Str, Ste = pr(Str), pr(Ste)
    T = Str.shape[2]; ed = np.linspace(0, T, K+1).astype(int)
    ftr, fte = [], []
    for i in range(K):
        a, b = Str[:, :, ed[i]:ed[i+1]], Ste[:, :, ed[i]:ed[i+1]]
        Ca, Cb = Covariances("oas").fit_transform(a), Covariances("oas").fit_transform(b)
        ts = TangentSpace(metric="riemann").fit(Ca)
        ftr.append(ts.transform(Ca)); fte.append(ts.transform(Cb))
    return np.hstack(ftr), np.hstack(fte)

def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5))
def lda(): return make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))

NAMES = ["volt", "xdawn", "winriem", "XDAWN+WINRIEM"]
RES = {"place": {k: [] for k in NAMES}, "ph11": {k: [] for k in NAMES}}
print("Xdawn (spatial SNR) + windowed Riemannian (temporal order) | place & 11-phoneme\n")
for f in FILES:
    Xall, ph = load(f)
    tasks = {}
    k = np.isin(ph, ["b", "p", "d", "t"])
    tasks["place"] = (Xall[k], np.where(np.isin(ph[k], ["d", "t"]), 1, 0))
    k2 = np.isin(ph, PHON)
    tasks["ph11"] = (Xall[k2], np.array([PHON.index(v) for v in ph[k2]]))
    for tname, (X, y) in tasks.items():
        acc = {k: [] for k in NAMES}
        for s in range(3):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                c = lda(); c.fit(X[tr][:, :, ::10].reshape(len(tr), -1), y[tr])
                acc["volt"].append(balanced_accuracy_score(y[te], c.predict(X[te][:, :, ::10].reshape(len(te), -1))))
                xd = Xdawn(nfilter=4, estimator="oas").fit(X[tr], y[tr])
                Ztr, Zte = xd.transform(X[tr]), xd.transform(X[te])
                c = lda(); c.fit(Ztr[:, :, ::10].reshape(len(tr), -1), y[tr])
                acc["xdawn"].append(balanced_accuracy_score(y[te], c.predict(Zte[:, :, ::10].reshape(len(te), -1))))
                a, b = win_riem(X[tr], X[te], K=4, npca=20)
                c = lr(); c.fit(a, y[tr]); acc["winriem"].append(balanced_accuracy_score(y[te], c.predict(b)))
                nf = min(Ztr.shape[1], 12)
                a, b = win_riem(Ztr[:, :nf], Zte[:, :nf], K=4)
                c = lr(); c.fit(a, y[tr]); acc["XDAWN+WINRIEM"].append(balanced_accuracy_score(y[te], c.predict(b)))
        for k_ in NAMES: RES[tname][k_].append(float(np.mean(acc[k_])))
    print("  %-8s place: %s | ph11: %s" % (os.path.basename(f).split("_")[0],
          " ".join("%.3f" % RES["place"][k_][-1] for k_ in NAMES),
          " ".join("%.3f" % RES["ph11"][k_][-1] for k_ in NAMES)), flush=True)

for tname, ch in [("place", 0.5), ("ph11", 1/11)]:
    print("\n=== %s (chance %.3f) ===" % (tname, ch))
    base = np.array(RES[tname]["volt"])
    for k_ in NAMES:
        a = np.array(RES[tname][k_]); _, p0 = ttest_1samp(a, ch); _, pb = ttest_rel(a, base)
        print("  %-15s %.3f  vs-chance p=%.5f  vs-volt %+.3f (p=%.4f)" % (k_, a.mean(), p0, a.mean()-base.mean(), pb))
