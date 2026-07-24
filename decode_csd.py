import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel
import mne
mne.set_log_level("ERROR")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score

P = os.path.expanduser("~/Downloads/EEG_Data_Processing_ses02_npz/EEG_Data_Processing_ses02_all.npz")
d = np.load(P, allow_pickle=True)
X = d["X"].astype(float)
t = d["times"]
sf = float(d["sfreq"])
ch = [str(c) for c in d["channel_names"]]
dsi = d["dataset_index_per_trial"]
dsubj = np.array([str(s) for s in d["dataset_subjects"]])
dcond = np.array([str(c) for c in d["dataset_conditions"]])
pos = d["dataset_channel_positions_xyz"]
lab = np.array([str(s) for s in d["event_label"]])

print("datasets:", len(dsubj), "| conditions:", sorted(set(dcond)))
pv = np.array([np.abs(pos[i] - pos[0]).max() for i in range(len(pos))])
print("max position deviation across datasets (m): %.4f  (0 = identical montage)" % pv.max())

def place_lab(labels: np.ndarray) -> np.ndarray:
    out = []
    for s in labels:
        tok = s.split("/")[0]
        out.append("a" if tok.endswith("a") else ("b" if tok.endswith("b") else "?"))
    return np.array(out)

pl = place_lab(lab)

def csd_transform(Xd: np.ndarray, positions: np.ndarray) -> np.ndarray:
    mont = mne.channels.make_dig_montage(ch_pos=dict(zip(ch, positions)), coord_frame="head")
    info = mne.create_info(ch, sf, "eeg")
    info.set_montage(mont)
    ep = mne.EpochsArray(Xd, info, tmin=float(t[0]))
    return mne.preprocessing.compute_current_source_density(ep).get_data()

CONTROL = "Control"
subj_data = {}
for ds in range(len(dsubj)):
    if dcond[ds] != CONTROL:
        continue
    m = (dsi == ds) & np.isin(pl, ["a", "b"])
    if m.sum() < 16:
        continue
    y = (pl[m] == "a").astype(int)
    if min(np.bincount(y)) < 6:
        continue
    Xd = X[m]
    try:
        Xc = csd_transform(Xd, pos[ds])
    except Exception as e:
        print("  CSD failed ds%d (%s): %s" % (ds, dsubj[ds], e)); continue
    subj_data[dsubj[ds]] = (Xd, Xc, y)

print("usable control subjects: %d | trials: %s\n" % (len(subj_data), [len(v[2]) for v in subj_data.values()]))

volt = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
def feat(Xa: np.ndarray, wmax: float) -> np.ndarray:
    w = (t >= 0.0) & (t <= wmax)
    return Xa[:, :, w][:, :, ::6].reshape(len(Xa), -1)

for wmax in [0.5, 1.0]:
    res = {"raw": [], "csd": []}
    for name, (Xr, Xc, y) in subj_data.items():
        for tag, Xa in [("raw", Xr), ("csd", Xc)]:
            F = feat(Xa, wmax)
            res[tag].append(np.mean([cross_val_score(volt, F, y, cv=StratifiedKFold(5, shuffle=True, random_state=s), scoring="balanced_accuracy").mean() for s in range(5)]))
    print("=== window 0-%.1fs | WITHIN-subject place ===" % wmax)
    for tag in ["raw", "csd"]:
        a = np.array(res[tag]); _, p = ttest_1samp(a, 0.5)
        print("  %-4s %.3f +- %.3f  (%d/%d>ch)  p=%.4f %s" % (tag, a.mean(), a.std(), int((a > 0.5).sum()), len(a), p, "**SIG**" if p < 0.05 and a.mean() > 0.5 else ""))
    _, pc = ttest_rel(res["csd"], res["raw"])
    print("  CSD-raw = %+.3f  paired-p=%.4f" % (np.mean(res["csd"]) - np.mean(res["raw"]), pc))

    names = list(subj_data.keys())
    for tag, ix in [("raw", 0), ("csd", 1)]:
        accs = []
        F_all = [(feat(subj_data[n][ix], wmax), subj_data[n][2]) for n in names]
        for i in range(len(names)):
            Ftr = np.concatenate([F_all[j][0] for j in range(len(names)) if j != i], 0)
            ytr = np.concatenate([F_all[j][1] for j in range(len(names)) if j != i], 0)
            clf = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
            clf.fit(Ftr, ytr)
            accs.append(balanced_accuracy_score(F_all[i][1], clf.predict(F_all[i][0])))
        a = np.array(accs); _, p = ttest_1samp(a, 0.5)
        print("  LOSO %-4s %.3f +- %.3f  (%d/%d>ch)  p=%.4f %s" % (tag, a.mean(), a.std(), int((a > 0.5).sum()), len(a), p, "**SIG**" if p < 0.05 and a.mean() > 0.5 else ""))
    print()
