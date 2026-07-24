import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
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
VOWELS = ["a", "e", "i", "o", "u"]
STOPS = ["b", "p", "d", "t"]

def load(sub):
    d = np.load("npz/sub-%s_ses-02.npz" % sub, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    meta = pd.DataFrame({c: [str(v) for v in d[c]] for c in ["task", "tms_target", "phoneme"]})
    return X, t, meta

def syl_vowels(sub):
    ev = pd.read_csv("events/sub-%s_phonemes_events.tsv" % sub, sep="\t")
    st = ev[ev.trial_type == "stimulus"].reset_index(drop=True)
    return st.phoneme1.astype(str).values, st.phoneme2.astype(str).str.strip().values

def cut(X, t, mask, t0=0.0, t1=0.5):
    bm = (t >= -0.2) & (t <= -0.11); wm = (t >= t0) & (t <= t1)
    Xm = X[mask]
    return (Xm - Xm[:, :, bm].mean(2, keepdims=True))[:, :, wm]

def cvd(Xw, y, seeds=5, folds=4, npca=20):
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 4: return np.nan
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

R = {k: [] for k in ["iso_vowel", "syl_vowel_CTRL", "syl_vowel_ALLTMS", "syl_place_CTRL", "iso_place"]}
print("RE-VERIFY with matched conditions (control-only vs all-TMS)\n")
print("%-5s %10s %12s %12s %12s %10s" % ("sub", "isoVowel", "sylVow_CTRL", "sylVow_ALL", "sylPlace_C", "isoPlace"))
for sub in SUBS:
    X, t, meta = load(sub)
    ph = meta.phoneme.values; task = meta.task.values
    ctrl = np.array([s.startswith("control") for s in meta.tms_target.values])
    m_iso_v = (task == "single-phoneme") & np.isin(ph, VOWELS) & ctrl
    yv = np.array([VOWELS.index(v) for v in ph[m_iso_v]])
    R["iso_vowel"].append(cvd(cut(X, t, m_iso_v), yv))
    m_iso_p = (task == "single-phoneme") & np.isin(ph, STOPS) & ctrl
    R["iso_place"].append(cvd(cut(X, t, m_iso_p), np.where(np.isin(ph[m_iso_p], ["d", "t"]), 1, 0)))
    try:
        p1, p2 = syl_vowels(sub)
    except Exception:
        for k in ["syl_vowel_CTRL", "syl_vowel_ALLTMS", "syl_place_CTRL"]: R[k].append(np.nan)
        continue
    m_all = task == "phonemes"
    if m_all.sum() != len(p1):
        for k in ["syl_vowel_CTRL", "syl_vowel_ALLTMS", "syl_place_CTRL"]: R[k].append(np.nan)
        continue
    ctrl_syl = ctrl[m_all]
    Xall = cut(X, t, m_all)
    ok = np.isin(p2, VOWELS)
    ya = np.array([VOWELS.index(v) if v in VOWELS else -1 for v in p2])
    R["syl_vowel_ALLTMS"].append(cvd(Xall[ok], ya[ok], seeds=3))
    sel = ok & ctrl_syl
    R["syl_vowel_CTRL"].append(cvd(Xall[sel], ya[sel], seeds=5, folds=3))
    selp = ctrl_syl & np.isin(p1, STOPS)
    R["syl_place_CTRL"].append(cvd(Xall[selp], np.where(np.isin(p1[selp], ["d", "t"]), 1, 0), seeds=5))
    print("%-5s %10.3f %12.3f %12.3f %12.3f %10.3f" % (
        sub, R["iso_vowel"][-1], R["syl_vowel_CTRL"][-1], R["syl_vowel_ALLTMS"][-1],
        R["syl_place_CTRL"][-1], R["iso_place"][-1]), flush=True)

print("\n=== GROUP ===")
for k, ch, lbl in [("iso_vowel", .2, "ISOLATED vowel 5-way (control)"),
                   ("syl_vowel_CTRL", .2, "SYLLABLE vowel 5-way, CONTROL only"),
                   ("syl_vowel_ALLTMS", .2, "SYLLABLE vowel 5-way, all TMS (flawed)"),
                   ("iso_place", .5, "ISOLATED place (control)"),
                   ("syl_place_CTRL", .5, "SYLLABLE place, control only")]:
    a = np.array([v for v in R[k] if not np.isnan(v)])
    if len(a) == 0: continue
    _, p = ttest_1samp(a, ch)
    print("  %-42s %.3f +- %.3f  (%d/%d)  p=%.5f %s" % (
        lbl, a.mean(), a.std(), int((a > ch).sum()), len(a), p, "**" if p < 0.05 and a.mean() > ch else ""))
