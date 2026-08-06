import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power as fmp
from scipy.stats import ttest_1samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score

FILES = [f for f in sorted(glob.glob("data/npz/sub-*.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
WMIN, WMAX = 0.0, 0.5
BMIN, BMAX = -0.2, -0.11

def ea_whiten(Xb: np.ndarray) -> np.ndarray:
    R = np.mean([x @ x.T / x.shape[1] for x in Xb], 0)
    W = np.real(fmp(R, -0.5))
    return np.stack([W @ x for x in Xb], 0)

def build(f: str, tasks: list[str]) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(f, allow_pickle=True); X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({c: d[c] for c in ["task", "phoneme"]})
    ph = df.phoneme.values
    mask = np.isin(df.task.values, tasks) & np.isin(ph, ["b", "p", "d", "t"])
    y = np.where(np.isin(ph[mask], ["d", "t"]), 1, 0)
    Xm = X[mask]
    bm = (t >= BMIN) & (t <= BMAX); wm = (t >= WMIN) & (t <= WMAX)
    Xb = (Xm - Xm[:, :, bm].mean(2, keepdims=True))[:, :, wm]
    return Xb, y

def loso(subjects: list[tuple[str, np.ndarray, np.ndarray]]) -> np.ndarray:
    accs = []
    for i in range(len(subjects)):
        Ftr = np.concatenate([subjects[j][1] for j in range(len(subjects)) if j != i], 0)
        ytr = np.concatenate([subjects[j][2] for j in range(len(subjects)) if j != i], 0)
        clf = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
        clf.fit(Ftr, ytr)
        accs.append(balanced_accuracy_score(subjects[i][2], clf.predict(subjects[i][1])))
    return np.array(accs)

for label, tasks, flist in [
        ("single-phoneme only (S*)", ["single-phoneme"], [f for f in FILES if "sub-S" in f]),
        ("all data (23 subj, single+phonemes)", ["single-phoneme", "phonemes"], FILES)]:
    print("\n=== PLACE-STOPS cross-subject | %s ===" % label)
    for mode in ["no-align", "EA-align"]:
        subj = []
        for f in flist:
            Xb, y = build(f, tasks)
            if len(np.unique(y)) < 2 or min(np.bincount(y)) < 10:
                continue
            Xa = ea_whiten(Xb) if mode == "EA-align" else Xb
            F = Xa[:, :, ::10].reshape(len(Xa), -1)
            subj.append((os.path.basename(f).split("_")[0], F, y))
        l = loso(subj)
        _, p = ttest_1samp(l, 0.5)
        print("  %-10s LOSO bal %.3f +- %.3f  (%d/%d>ch)  p=%.4f  %s" % (
            mode, l.mean(), l.std(), int((l > 0.5).sum()), len(l), p, "**SIG**" if p < 0.05 and l.mean() > 0.5 else ""))
