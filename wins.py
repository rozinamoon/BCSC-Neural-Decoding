import warnings
warnings.filterwarnings("ignore")
import glob
import os
import numpy as np
from multiprocessing import Pool
from scipy.stats import ttest_rel, t as tdist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

PHON = ["a", "i", "e", "u", "o", "s", "z", "p", "b", "t", "d"]
STOPS = ["b", "p", "d", "t"]
NPCA = {"place": 40, "ph11": 25}
CHANCE = {"place": 0.5, "ph11": 1 / 11}
WINS = {"early_0_500": (0.0, 0.5, {"place": 1, "ph11": 3}),
        "late_500_1000": (0.5, 1.1, {"place": 1, "ph11": 3}),
        "full_0_1000": (0.0, 1.1, {"place": 2, "ph11": 6})}
OLD = {"ph11": 0.169, "place": 0.648}


def load(f):
    d = np.load(f, allow_pickle=True)
    X, t = d["X"].astype(float), d["times"]
    k = (d["task"].astype(str) == "single_phoneme") & (d["tms_target"].astype(str) == "control")
    X = X[k] - X[k][:, :, t <= 0.02].mean(2, keepdims=True)
    return X, t, d["phoneme1"].astype(str)[k]


def ev(Xtr, ytr, Xte, yte, K, npca):
    pc = PCA(min(npca, len(Xtr) - 2)).fit(Xtr.transpose(0, 2, 1).reshape(-1, Xtr.shape[1]))
    pr = lambda Z: np.stack([pc.transform(z.T).T for z in Z], 0)
    A, B = pr(Xtr), pr(Xte)
    ed = np.linspace(0, A.shape[2], K + 1).astype(int)
    va, vb = [], []
    for i in range(K):
        Ca = Covariances("oas").fit_transform(A[:, :, ed[i]:ed[i + 1]])
        Cb = Covariances("oas").fit_transform(B[:, :, ed[i]:ed[i + 1]])
        ts = TangentSpace(metric="riemann").fit(Ca)
        va.append(ts.transform(Ca))
        vb.append(ts.transform(Cb))
    a, b = np.concatenate(va, 1), np.concatenate(vb, 1)
    m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")).fit(a, ytr)
    return balanced_accuracy_score(yte, m.predict(b))


def one(f):
    X, t, ph = load(f)
    out = {}
    for tn, keep in [("place", STOPS), ("ph11", PHON)]:
        m = np.isin(ph, keep)
        p = ph[m]
        y = (np.where(np.isin(p, ["d", "t"]), 1, 0) if tn == "place"
             else np.array([PHON.index(v) for v in p]))
        for wn, (t0, t1, KK) in WINS.items():
            msk = (t >= t0) & (t < t1)
            Xs = X[m][:, :, msk]
            K = KK[tn]
            acc = []
            for sd in range(3):
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(Xs, y):
                    acc.append(ev(Xs[tr], y[tr], Xs[te], y[te], K, NPCA[tn]))
            out[(tn, wn)] = (float(np.mean(acc)), int(msk.sum()), int(msk.sum() // K))
    return os.path.basename(f)[:3], out


if __name__ == "__main__":
    FILES = sorted(glob.glob("npz2/*.npz"))
    print("WINDOW DECOMPOSITION on npz2 | %d subj | samples/window matched ~42\n" % len(FILES),
          flush=True)
    res = []
    with Pool(16) as pool:
        for sub, o in pool.imap_unordered(one, FILES):
            res.append((sub, o))
            print("  %s  ph11 early=%.3f late=%.3f full=%.3f | place early=%.3f late=%.3f full=%.3f"
                  % (sub, o[("ph11", "early_0_500")][0], o[("ph11", "late_500_1000")][0],
                     o[("ph11", "full_0_1000")][0], o[("place", "early_0_500")][0],
                     o[("place", "late_500_1000")][0], o[("place", "full_0_1000")][0]),
                  flush=True)

    def ci(d):
        m, s, n = d.mean(), d.std(ddof=1), len(d)
        h = tdist.ppf(.975, n - 1) * s / np.sqrt(n)
        return m, m - h, m + h

    for tn in ["place", "ph11"]:
        print("\n=== %s (chance %.3f) | old-data 0-500ms reference = %.3f ==="
              % (tn, CHANCE[tn], OLD[tn]))
        V = {wn: np.array([o[(tn, wn)][0] for _, o in res]) for wn in WINS}
        for wn in WINS:
            n, sw = res[0][1][(tn, wn)][1:]
            print("  %-15s %.3f   (%d samp, K=%d, %d samp/win)"
                  % (wn, V[wn].mean(), n, WINS[wn][2][tn], sw))
        m, lo, hi = ci(V["full_0_1000"] - V["early_0_500"])
        print("  full - early    %+.3f  95%%CI [%+.3f,%+.3f]  p=%.4f   <- WINDOW effect"
              % (m, lo, hi, ttest_rel(V["full_0_1000"], V["early_0_500"]).pvalue))
        print("  early - old     %+.3f                                  <- ICA effect"
              % (V["early_0_500"].mean() - OLD[tn]))
        m, lo, hi = ci(V["late_500_1000"] - CHANCE[tn])
        print("  late vs chance  %+.3f  95%%CI [%+.3f,%+.3f]" % (m, lo, hi))
