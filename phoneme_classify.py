import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

FILES = [f for f in sorted(glob.glob("npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
D_PCA = 20
PHONEMES = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
VOWELS = ["a", "i", "e", "u", "o"]

def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "tms_target", "phoneme"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    return (X[m] - X[m][:, :, b].mean(2, keepdims=True))[:, :, w], df.phoneme.values[m]

def riemann_feats(Xtr, Xte):
    p = PCA(D_PCA).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    def proj(Z): return np.stack([p.transform(z.T).T for z in Z], 0)
    Ctr = Covariances("oas").fit_transform(proj(Xtr))
    Cte = Covariances("oas").fit_transform(proj(Xte))
    ts = TangentSpace(metric="riemann").fit(Ctr)
    return ts.transform(Ctr), ts.transform(Cte)

def run_cv(X, y, method, seeds=3):
    accs, preds, trues = [], [], []
    for s in range(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            if method == "riemann":
                Ftr, Fte = riemann_feats(X[tr], X[te])
                c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5))
            else:
                Ftr = X[tr][:, :, ::10].reshape(len(tr), -1); Fte = X[te][:, :, ::10].reshape(len(te), -1)
                c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
            c.fit(Ftr, y[tr]); pr = c.predict(Fte)
            accs.append(balanced_accuracy_score(y[te], pr))
            if s == 0:
                preds.append(pr); trues.append(y[te])
    return float(np.mean(accs)), np.concatenate(preds), np.concatenate(trues)

R = {"ph11_riem": [], "ph11_volt": [], "vow5": [], "cons6": []}
CM = np.zeros((11, 11))
print("PHONEME CLASSIFICATION (final goal) | 11-class, chance=%.3f\n" % (1/11))
for f in FILES:
    Xm, ph = load(f)
    keep = np.isin(ph, PHONEMES)
    X, p_ = Xm[keep], ph[keep]
    y = np.array([PHONEMES.index(v) for v in p_])
    a_r, pr, tr_ = run_cv(X, y, "riemann")
    a_v, _, _ = run_cv(X, y, "volt")
    R["ph11_riem"].append(a_r); R["ph11_volt"].append(a_v)
    CM += confusion_matrix(tr_, pr, labels=range(11))
    vm = np.isin(p_, VOWELS)
    yv = np.array([VOWELS.index(v) for v in p_[vm]])
    cm_ = ~vm
    cons = [c for c in PHONEMES if c not in VOWELS]
    yc = np.array([cons.index(v) for v in p_[cm_]])
    R["vow5"].append(run_cv(X[vm], yv, "riemann")[0])
    R["cons6"].append(run_cv(X[cm_], yc, "riemann")[0])
    print("  %-8s 11-class: riem %.3f volt %.3f | vowel5 %.3f cons6 %.3f" % (
        os.path.basename(f).split("_")[0], a_r, a_v, R["vow5"][-1], R["cons6"][-1]), flush=True)

print("\n=== GROUP ===")
for k, ch, lbl in [("ph11_riem", 1/11, "11-phoneme Riemannian"), ("ph11_volt", 1/11, "11-phoneme voltage-LDA"),
                   ("vow5", 1/5, "5-vowel Riemannian"), ("cons6", 1/6, "6-consonant Riemannian")]:
    a = np.array(R[k]); _, p = ttest_1samp(a, ch)
    print("  %-24s %.3f +- %.3f  (chance %.3f)  %d/%d>ch  p=%.5f %s" % (
        lbl, a.mean(), a.std(), ch, int((a > ch).sum()), len(a), p, "**SIG**" if p < 0.05 and a.mean() > ch else ""))

print("\n=== CONFUSION STRUCTURE (row-normalised, pooled) ===")
CMn = CM / CM.sum(1, keepdims=True)
print("      " + "  ".join("%4s" % p for p in PHONEMES))
for i, p in enumerate(PHONEMES):
    print("  %2s  " % p + "  ".join("%.2f" % v for v in CMn[i]))
vi = [PHONEMES.index(v) for v in VOWELS]; ci = [i for i in range(11) if i not in vi]
print("\n  vowel->vowel  %.3f | vowel->cons %.3f" % (CMn[np.ix_(vi, vi)].sum()/len(vi), CMn[np.ix_(vi, ci)].sum()/len(vi)))
print("  cons->cons    %.3f | cons->vowel %.3f" % (CMn[np.ix_(ci, ci)].sum()/len(ci), CMn[np.ix_(ci, vi)].sum()/len(ci)))
