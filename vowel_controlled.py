import os
import subprocess
import glob
import numpy as np
import pandas as pd
import collections
from scipy.stats import ttest_1samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

SUBS = ["S01", "S02", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]
BASE = "https://s3.amazonaws.com/openneuro.org/ds006104"
OUT = "events"
STOPS = ["b", "p", "d", "t"]
VOWELS = ["a", "e", "i", "o", "u"]
os.makedirs(OUT, exist_ok=True)

for s in SUBS:
    dst = os.path.join(OUT, "sub-%s_singlephoneme_events.tsv" % s)
    if not os.path.exists(dst):
        url = "%s/sub-%s/ses-02/eeg/sub-%s_ses-02_task-singlephoneme_events.tsv" % (BASE, s, s)
        subprocess.run(["curl", "-s", "--max-time", "60", "-o", dst, url], capture_output=True)

def vowel_map(sub):
    f = os.path.join(OUT, "sub-%s_phonemes_events.tsv" % sub)
    df = pd.read_csv(f, sep="\t")
    st = df[df.trial_type == "stimulus"].reset_index(drop=True)
    return st.phoneme1.astype(str).values, st.phoneme2.astype(str).str.strip().values

def load_npz(sub):
    d = np.load("npz/sub-%s_ses-02.npz" % sub, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    return X, t, meta

def cut(X, t, mask, t0=0.0, t1=0.5):
    bm = (t >= -0.2) & (t <= -0.11); wm = (t >= t0) & (t <= t1)
    Xm = X[mask]
    return (Xm - Xm[:, :, bm].mean(2, keepdims=True))[:, :, wm]

def cvd(Xw, y, npca=20, seeds=4, folds=4):
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5: return np.nan
    acc = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=s).split(Xw, y):
            pc = PCA(min(npca, len(tr)-2)).fit(Xw[tr].transpose(0, 2, 1).reshape(-1, Xw.shape[1]))
            prj = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
            A, B = prj(Xw[tr]), prj(Xw[te])
            T = A.shape[2]; ed = np.linspace(0, T, 3).astype(int)
            fa, fb = [], []
            for i in range(2):
                Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
                Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
                ts = TangentSpace(metric="riemann").fit(Ca)
                fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
            c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
            c.fit(np.hstack(fa), y[tr]); acc.append(balanced_accuracy_score(y[te], c.predict(np.hstack(fb))))
    return float(np.mean(acc))

R = {k: [] for k in ["all_mixed", "WITHIN_VOWEL", "cross_vowel", "vowel_decode"]}
print("VOWEL-CONTEXT CONTROLLED place decoding in CV syllables\n")
for sub in SUBS:
    try:
        p1, p2 = vowel_map(sub)
    except Exception as e:
        print("  %s events missing" % sub); continue
    X, t, meta = load_npz(sub)
    m_syl = (meta.task.values == "phonemes")
    if m_syl.sum() != len(p1):
        print("  %s length mismatch npz=%d ev=%d" % (sub, m_syl.sum(), len(p1))); continue
    Xs = cut(X, t, m_syl)
    keep = np.isin(p1, STOPS) & np.isin(p2, VOWELS)
    Xs, c1, v1 = Xs[keep], p1[keep], p2[keep]
    y = np.where(np.isin(c1, ["d", "t"]), 1, 0)
    R["all_mixed"].append(cvd(Xs, y))
    per_v = []
    for v in VOWELS:
        mv = v1 == v
        if mv.sum() >= 20:
            per_v.append(cvd(Xs[mv], y[mv], seeds=5, folds=3))
    R["WITHIN_VOWEL"].append(np.nanmean(per_v) if per_v else np.nan)
    tr_v = np.isin(v1, VOWELS[:3]); te_v = ~tr_v
    if min(tr_v.sum(), te_v.sum()) > 20:
        pc = PCA(20).fit(Xs[tr_v].transpose(0, 2, 1).reshape(-1, Xs.shape[1]))
        prj = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
        A, B = prj(Xs[tr_v]), prj(Xs[te_v])
        T = A.shape[2]; ed = np.linspace(0, T, 3).astype(int)
        fa, fb = [], []
        for i in range(2):
            Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i+1]])
            Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i+1]])
            ts = TangentSpace(metric="riemann").fit(Ca)
            fa.append(ts.transform(Ca)); fb.append(ts.transform(Cb))
        c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
        c.fit(np.hstack(fa), y[tr_v])
        R["cross_vowel"].append(balanced_accuracy_score(y[te_v], c.predict(np.hstack(fb))))
    else:
        R["cross_vowel"].append(np.nan)
    yv = np.array([VOWELS.index(v) for v in v1])
    R["vowel_decode"].append(cvd(Xs, yv, seeds=2))
    print("  %-5s mixed %.3f | WITHIN-VOWEL %.3f | cross-vowel %.3f | VOWEL(5way,ch=.2) %.3f" % (
        sub, R["all_mixed"][-1], R["WITHIN_VOWEL"][-1], R["cross_vowel"][-1], R["vowel_decode"][-1]), flush=True)

print("\n=== GROUP ===")
for k, ch, lbl in [("all_mixed", .5, "place, vowel contexts MIXED"),
                   ("WITHIN_VOWEL", .5, "place, WITHIN each vowel context"),
                   ("cross_vowel", .5, "place, train a/e/i -> test o/u"),
                   ("vowel_decode", .2, "vowel identity 5-way (positive ctrl)")]:
    a = np.array([v for v in R[k] if not np.isnan(v)])
    if len(a) == 0: continue
    _, p = ttest_1samp(a, ch)
    print("  %-40s %.3f +- %.3f  (%d/%d)  p=%.5f %s" % (
        lbl, a.mean(), a.std(), int((a > ch).sum()), len(a), p, "**" if p < 0.05 and a.mean() > ch else ""))
