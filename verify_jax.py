import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import jax.numpy as jnp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from jax_pipeline import windowed_riemann, shrinkage_lda_fit, predict, balanced_accuracy, standardise

STOPS = ["b", "p", "d", "t"]

d = np.load("npz/sub-S04_ses-02.npz", allow_pickle=True)
X, t = d["X"].astype(float), d["times"]
meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
k = ((meta.task == "single-phoneme") & meta.tms_target.str.startswith("control")).values
bm = (t >= -0.2) & (t <= -0.11); wm = (t >= 0.0) & (t <= 0.5)
Xk = X[k]
Xb = (Xk - Xk[:, :, bm].mean(2, keepdims=True))[:, :, wm]
ph = meta.phoneme.values[k]
m = np.isin(ph, STOPS)
Xs, y = Xb[m], np.where(np.isin(ph[m], ["d", "t"]), 1, 0)

def np_feats(Xtr, Xte, K=2, npca=20):
    pc = PCA(npca).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    pr = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = pr(Xtr), pr(Xte)
    ed = np.linspace(0, A.shape[2], K+1).astype(int)
    fa, fb = [], []
    for i in range(K):
        Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
        Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
    return np.hstack(fa), np.hstack(fb)

tr, te = next(iter(StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, y)))
Fnp_tr, Fnp_te = np_feats(Xs[tr], Xs[te])
Fjx_tr, Fjx_te = windowed_riemann(jnp.asarray(Xs[tr]), jnp.asarray(Xs[te]), 2, 20)
Fjx_tr, Fjx_te = np.asarray(Fjx_tr), np.asarray(Fjx_te)

print("=== FEATURE COMPARISON (S04, fold 0) ===")
print("  shapes np=%s jax=%s" % (Fnp_tr.shape, Fjx_tr.shape))
print("  np   : mean %+.4f  std %.4f  |max| %.3f" % (Fnp_tr.mean(), Fnp_tr.std(), np.abs(Fnp_tr).max()))
print("  jax  : mean %+.4f  std %.4f  |max| %.3f" % (Fjx_tr.mean(), Fjx_tr.std(), np.abs(Fjx_tr).max()))
cc = [abs(np.corrcoef(Fnp_tr[:, i], Fjx_tr[:, i])[0, 1]) for i in range(min(Fnp_tr.shape[1], Fjx_tr.shape[1]))]
print("  |corr| per-dim: median %.3f  min %.3f  frac>0.99 %.2f" % (
    np.median(cc), np.min(cc), np.mean(np.array(cc) > 0.99)))

print("\n=== CLASSIFIER COMPARISON (identical features) ===")
for nm, Ftr, Fte in [("np-feats", Fnp_tr, Fnp_te), ("jax-feats", Fjx_tr, Fjx_te)]:
    c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    c.fit(Ftr, y[tr]); a_sk = balanced_accuracy_score(y[te], c.predict(Fte))
    A, B = standardise(jnp.asarray(Ftr), jnp.asarray(Fte))
    Wc, b = shrinkage_lda_fit(A, jnp.asarray(y[tr]), 2)
    a_jx = float(balanced_accuracy(jnp.asarray(y[te]), predict(B, Wc, b), 2))
    print("  %-10s sklearn-LDA %.3f | jax-LDA %.3f" % (nm, a_sk, a_jx))

print("\n=== jax LDA shrinkage sweep (on np features) ===")
A, B = standardise(jnp.asarray(Fnp_tr), jnp.asarray(Fnp_te))
for sh in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]:
    Wc, b = shrinkage_lda_fit(A, jnp.asarray(y[tr]), 2, sh)
    print("  shrink %.2f : %.3f" % (sh, float(balanced_accuracy(jnp.asarray(y[te]), predict(B, Wc, b), 2))))
