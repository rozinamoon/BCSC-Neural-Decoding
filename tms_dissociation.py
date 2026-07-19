import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

FILES = [f for f in sorted(glob.glob("data/npz/sub-S*_ses-02.npz"))
         if np.load(f, allow_pickle=True)["X"].shape[1] == 61]
BMIN, BMAX = -0.2, -0.11
WMIN, WMAX = 0.0, 0.5
volt = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))

def per_class_recall(F: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rec = {0: [], 1: []}
    for seed in range(5):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(F, y):
            volt.fit(F[tr], y[tr]); pred = volt.predict(F[te])
            for cls in [0, 1]:
                mm = y[te] == cls
                if mm.sum() > 0:
                    rec[cls].append((pred[mm] == cls).mean())
    return float(np.mean(rec[1])), float(np.mean(rec[0]))

def feats(X, t, mask):
    bmask = (t >= BMIN) & (t <= BMAX); wmask = (t >= WMIN) & (t <= WMAX)
    Xc = X[mask]
    Xb = (Xc - Xc[:, :, bmask].mean(2, keepdims=True))[:, :, wmask]
    return Xb[:, :, ::10].reshape(len(Xb), -1)

inter, bil_l, bil_t, alv_l, alv_t = [], [], [], [], []
STOPS_ONLY = False
print("D'Ausilio double-dissociation: congruent TMS facilitates congruent place")
print("interaction = [bil(lip)-bil(tongue)] - [alv(lip)-alv(tongue)] ; >0 = dissociation\n")
print("%-8s %8s %8s %8s %8s %10s" % ("subj", "bilLip", "bilTon", "alvLip", "alvTon", "interact"))
for f in FILES:
    sub = os.path.basename(f).split("_")[0]
    d = np.load(f, allow_pickle=True); X, t = d["X"].astype(float), d["times"]
    df = pd.DataFrame({k: d[k] for k in ["task", "tms_target", "place", "phoneme"]})
    ph = (df.task == "phonemes").values
    plset = ["b", "p", "d", "t"] if STOPS_ONLY else ["alveolar", "bilabial"]
    def cond_feats(tgt):
        if STOPS_ONLY:
            cm = ph & (df.tms_target.values == tgt) & np.isin(df.phoneme.values, plset)
            y = np.isin(df.phoneme.values[cm], ["d", "t"]).astype(int)
        else:
            cm = ph & (df.tms_target.values == tgt) & np.isin(df.place.values, plset)
            y = (df.place.values[cm] == "alveolar").astype(int)
        return feats(X, t, cm), y
    Fl, yl = cond_feats("lip"); Ft, yt = cond_feats("tongue")
    if min(np.bincount(yl)) < 8 or min(np.bincount(yt)) < 8:
        print("%-8s skip (too few)" % sub); continue
    al_lip, bl_lip = per_class_recall(Fl, yl)
    al_ton, bl_ton = per_class_recall(Ft, yt)
    it = (bl_lip - bl_ton) - (al_lip - al_ton)
    inter.append(it); bil_l.append(bl_lip); bil_t.append(bl_ton); alv_l.append(al_lip); alv_t.append(al_ton)
    print("%-8s %8.3f %8.3f %8.3f %8.3f %10.3f" % (sub, bl_lip, bl_ton, al_lip, al_ton, it), flush=True)

inter = np.array(inter)
print("\n=== GROUP double-dissociation interaction ===")
_, p = ttest_1samp(inter, 0.0)
print("  interaction mean %+.4f +- %.4f  (n=%d)  vs-0 p=%.4f  %s" % (
    inter.mean(), inter.std(), len(inter), p, "**DISSOCIATION**" if p < 0.05 and inter.mean() > 0 else "ns"))
print("  congruent: bil under lip %.3f vs tongue %.3f (%+.3f, p=%.3f)" % (
    np.mean(bil_l), np.mean(bil_t), np.mean(bil_l)-np.mean(bil_t), ttest_rel(bil_l, bil_t)[1]))
print("  congruent: alv under tongue %.3f vs lip %.3f (%+.3f, p=%.3f)" % (
    np.mean(alv_t), np.mean(alv_l), np.mean(alv_t)-np.mean(alv_l), ttest_rel(alv_t, alv_l)[1]))
