import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.spatialfilters import Xdawn

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX, SF = 0.0, 0.5, 500.0
D_PCA, N_RES, LEAK, SUB = 20, 200, 0.3, 20
BANDS = [(1, 4), (4, 8), (8, 15), (15, 30)]

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w]
    ph = df.phoneme.values[m]
    k = np.isin(ph, ["b", "p", "d", "t"])
    return Xb[k], np.where(np.isin(ph[k], ["d", "t"]), 1, 0)

def esn_states(X, W, Win):
    p = PCA(D_PCA).fit(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
    U = np.stack([p.transform(x.T) for x in X], 0)
    U = (U - U.mean((0, 1))) / (U.std((0, 1)) + 1e-9)
    n, T, _ = U.shape; h = np.zeros((n, N_RES)); out = []
    for ti in range(T):
        h = (1-LEAK)*h + LEAK*np.tanh(h @ W.T + U[:, ti, :] @ Win.T)
        out.append(h.copy())
    return np.stack(out, 2)

def make_esn(seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.1)
    W = W / np.max(np.abs(np.linalg.eigvals(W)))
    return W, rng.normal(0, 1, (N_RES, D_IN := D_PCA)) * 0.5

def tangent_pipeline(Xtr, Xte, npca=D_PCA):
    p = PCA(npca).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    pr = lambda Z: np.stack([p.transform(z.T).T for z in Z], 0)
    Ctr = Covariances("oas").fit_transform(pr(Xtr)); Cte = Covariances("oas").fit_transform(pr(Xte))
    ts = TangentSpace(metric="riemann").fit(Ctr)
    return ts.transform(Ctr), ts.transform(Cte)

def csp_feats(Xtr, ytr, Xte, ncomp=6):
    def cov(Z):
        return np.stack([(z @ z.T)/(z.shape[1]-1) + 1e-8*np.eye(z.shape[0]) for z in Z], 0)
    c1 = cov(Xtr[ytr == 0]).mean(0); c2 = cov(Xtr[ytr == 1]).mean(0)
    ev, V = eigh(c1, c1 + c2)
    idx = np.argsort(np.abs(ev - 0.5))[::-1][:ncomp]; Wc = V[:, idx]
    f = lambda Z: np.log(np.einsum("ij,njk->nik", Wc.T, Z).var(2) + 1e-12)
    return f(Xtr), f(Xte)

def bandpass(X, lo, hi):
    b, a = butter(4, [lo/(SF/2), hi/(SF/2)], btype="band")
    return filtfilt(b, a, X, axis=2)

def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
def lda(): return make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))

W_esn, Win_esn = make_esn(0)
NAMES = ["riem_raw", "reservoir_flat", "RESERVOIR+RIEM", "xdawn_filt", "xdawn_cov", "csp", "fbank_riem"]
R = {k: [] for k in NAMES}
print("Filters & combinations | place-stops | which representation extracts the signal best?\n")
for f in FILES:
    X, y = load(f)
    Sres = esn_states(X, W_esn, Win_esn)
    Xb_bands = [bandpass(X, lo, hi) for lo, hi in BANDS]
    acc = {k: [] for k in NAMES}
    for s in range(3):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            Ftr, Fte = tangent_pipeline(X[tr], X[te])
            c = lr(); c.fit(Ftr, y[tr]); acc["riem_raw"].append(balanced_accuracy_score(y[te], c.predict(Fte)))
            Ftr = Sres[tr][:, :, ::SUB].reshape(len(tr), -1); Fte = Sres[te][:, :, ::SUB].reshape(len(te), -1)
            c = lda(); c.fit(Ftr, y[tr]); acc["reservoir_flat"].append(balanced_accuracy_score(y[te], c.predict(Fte)))
            Ftr, Fte = tangent_pipeline(Sres[tr], Sres[te])
            c = lr(); c.fit(Ftr, y[tr]); acc["RESERVOIR+RIEM"].append(balanced_accuracy_score(y[te], c.predict(Fte)))
            xd = Xdawn(nfilter=4, estimator="oas").fit(X[tr], y[tr])
            Ftr = xd.transform(X[tr])[:, :, ::10].reshape(len(tr), -1)
            Fte = xd.transform(X[te])[:, :, ::10].reshape(len(te), -1)
            c = lda(); c.fit(Ftr, y[tr]); acc["xdawn_filt"].append(balanced_accuracy_score(y[te], c.predict(Fte)))
            xc = make_pipeline(XdawnCovariances(nfilter=4, estimator="oas", xdawn_estimator="oas"),
                               TangentSpace(metric="riemann"), StandardScaler(),
                               LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
            xc.fit(X[tr], y[tr]); acc["xdawn_cov"].append(balanced_accuracy_score(y[te], xc.predict(X[te])))
            Ftr, Fte = csp_feats(X[tr], y[tr], X[te])
            c = lda(); c.fit(Ftr, y[tr]); acc["csp"].append(balanced_accuracy_score(y[te], c.predict(Fte)))
            fb_tr, fb_te = [], []
            for Xbnd in Xb_bands:
                a_, b_ = tangent_pipeline(Xbnd[tr], Xbnd[te], npca=12)
                fb_tr.append(a_); fb_te.append(b_)
            c = lr(); c.fit(np.hstack(fb_tr), y[tr])
            acc["fbank_riem"].append(balanced_accuracy_score(y[te], c.predict(np.hstack(fb_te))))
    for k in NAMES: R[k].append(float(np.mean(acc[k])))
    print("  %-8s " % os.path.basename(f).split("_")[0] + " ".join("%s %.3f" % (k[:6], R[k][-1]) for k in NAMES), flush=True)

base = np.array(R["riem_raw"])
print("\n=== GROUP (place-stops) ===")
for k in NAMES:
    a = np.array(R[k]); _, p0 = ttest_1samp(a, 0.5); _, pb = ttest_rel(a, base)
    print("  %-16s %.3f  vs-chance p=%.4f  vs-riem_raw %+.3f (p=%.4f)" % (k, a.mean(), p0, a.mean()-base.mean(), pb))
