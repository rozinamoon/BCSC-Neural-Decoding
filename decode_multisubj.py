import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score

FILES = sorted(glob.glob("data/npz/sub-S*_ses-02.npz"))
WMIN, WMAX = 0.0, 0.5

def band_power(x: np.ndarray, sf: float = 500.0) -> np.ndarray:
    from scipy.signal import welch
    n, c, T = x.shape
    f, P = welch(x, fs=sf, nperseg=min(128, T), axis=2)
    out = []
    for lo, hi in [(1, 4), (4, 8), (8, 13), (13, 30), (30, 60), (60, 120)]:
        s = (f >= lo) & (f < hi)
        out.append(np.log(P[:, :, s].mean(2) + 1e-20))
    return np.stack(out, 2).reshape(n, -1)

volt = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale", class_weight="balanced"))

def decode(F: np.ndarray, y: np.ndarray, est) -> tuple[float, float, float]:
    yy = pd.Categorical(y).codes
    accs = [cross_val_score(est, F, yy, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                            scoring="balanced_accuracy").mean() for s in range(5)]
    _, _, p = permutation_test_score(est, F, yy, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                     scoring="balanced_accuracy", n_permutations=200, random_state=0)
    return float(np.mean(accs)), float(np.std(accs)), float(p)

print("single-phoneme, control only, window %.1f-%.1fs, baseline-corr" % (WMIN, WMAX))
print("Voltage+shrinkLDA and BandPower+SVM (fast scan across subjects)\n")
res = {"category": [], "place": []}
for f in FILES:
    sub = os.path.basename(f).split("_")[0]
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    if X.shape[1] != 61:
        print(sub, "SKIP (%d ch)" % X.shape[1]); continue
    df = pd.DataFrame({k: d[k] for k in ["task", "tms_target", "category", "place"]})
    m = ((df.task == "single-phoneme") & (df.tms_target.str.startswith("control"))).values
    Xs, ds = X[m], df[m].reset_index(drop=True)
    b = t < 0; w = (t >= WMIN) & (t <= WMAX)
    Xb = (Xs - Xs[:, :, b].mean(2, keepdims=True))[:, :, w]
    Fv = Xb[:, :, ::5].reshape(len(Xb), -1)
    Fb = band_power(Xb)
    cat = ds.category.values
    cons = (ds.category != "vowels").values
    for tag, F in [("V", Fv), ("BP", Fb)]:
        vc = decode(F, cat, volt if tag == "V" else svm)
        pl = decode(F[cons], ds[cons].place.values, volt if tag == "V" else svm)
        if tag == "V":
            res["category"].append(vc[0]); res["place"].append(pl[0])
        print("  %s [%s]  category %.3f (p=%.3f%s)  place %.3f (p=%.3f%s)" % (
            sub, tag, vc[0], vc[2], "*" if vc[2] < 0.05 else "",
            pl[0], pl[2], "*" if pl[2] < 0.05 else ""))

print("\n=== GROUP (Voltage) ===")
for k in ["category", "place"]:
    a = np.array(res[k])
    ch = 0.5
    from scipy.stats import ttest_1samp
    tstat, pg = ttest_1samp(a, ch)
    print("  %-9s mean %.3f +- %.3f  (%d/%d subj >chance)  group-t p=%.4f %s" % (
        k, a.mean(), a.std(), int((a > ch).sum()), len(a), pg, "**SIG**" if pg < 0.05 and a.mean() > ch else ""))
