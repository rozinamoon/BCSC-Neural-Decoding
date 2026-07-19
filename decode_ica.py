import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score

DIR = "EEG_Data_Processing_ses02_npz copy/per_subject_condition"

def place_from_label(labels: np.ndarray) -> np.ndarray:
    out = []
    for s in labels:
        tok = str(s).split("/")[0]
        out.append("a" if tok.endswith("a") else ("b" if tok.endswith("b") else "?"))
    return np.array(out)

def load_control() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    subj = []
    for f in sorted(glob.glob(os.path.join(DIR, "*_c.npz"))):
        d = np.load(f, allow_pickle=True)
        X, t = d["X"].astype(float), d["times"]
        pl = place_from_label(d["event_label"])
        m = np.isin(pl, ["a", "b"])
        if m.sum() >= 16 and X.shape[1] == 61:
            y = (pl[m] == "a").astype(int)
            if min(np.bincount(y)) >= 6:
                subj.append((str(d["subject"]) if "subject" in d else os.path.basename(f)[:3], X[m], y, t))
    return subj

volt = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))

def feats(X: np.ndarray, t: np.ndarray, wmax: float) -> np.ndarray:
    w = (t >= 0.0) & (t <= wmax)
    return X[:, :, w][:, :, ::6].reshape(len(X), -1)

subj = load_control()
print("ICA-cleaned control data | %d subjects | place (a vs b)\n" % len(subj))
print("trials/subj:", [len(y) for _, _, y, _ in subj])

for wmax in [0.5, 1.0]:
    within = []
    for name, X, y, t in subj:
        F = feats(X, t, wmax)
        a = np.mean([cross_val_score(volt, F, y, cv=StratifiedKFold(5, shuffle=True, random_state=s), scoring="balanced_accuracy").mean() for s in range(5)])
        within.append(a)
    w = np.array(within); _, p = ttest_1samp(w, 0.5)
    print("\nwindow 0-%.1fs | WITHIN-subject place: %.3f +- %.3f  (%d/%d>ch)  p=%.4f %s" % (
        wmax, w.mean(), w.std(), int((w > 0.5).sum()), len(w), p, "**SIG**" if p < 0.05 and w.mean() > 0.5 else ""))

    F_all = [(feats(X, t, wmax), y) for _, X, y, t in subj]
    loso = []
    for i in range(len(subj)):
        Ftr = np.concatenate([F_all[j][0] for j in range(len(subj)) if j != i], 0)
        ytr = np.concatenate([F_all[j][1] for j in range(len(subj)) if j != i], 0)
        clf = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
        clf.fit(Ftr, ytr)
        loso.append(balanced_accuracy_score(F_all[i][1], clf.predict(F_all[i][0])))
    l = np.array(loso); _, p = ttest_1samp(l, 0.5)
    print("window 0-%.1fs | CROSS-subject LOSO: %.3f +- %.3f  (%d/%d>ch)  p=%.4f %s" % (
        wmax, l.mean(), l.std(), int((l > 0.5).sum()), len(l), p, "**SIG**" if p < 0.05 and l.mean() > 0.5 else ""))
