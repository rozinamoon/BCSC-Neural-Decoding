import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import fastriem as fr

rng = np.random.default_rng(0)
for n, d, T in [(180, 25, 42), (105, 40, 128)]:
    A = rng.normal(0, 1, (n, d, T)) + rng.normal(0, 1, (1, d, 1)) * 0.5
    t0 = time.time()
    Cp = Covariances("oas").fit_transform(A)
    ts = TangentSpace(metric="riemann").fit(Cp)
    Vp = ts.transform(Cp)
    t1 = time.time()
    Cf = fr.cov_oas(A)
    Mf = fr.riemann_mean(Cf)
    Vf = fr.tangent(Cf, Mf)
    t2 = time.time()
    dc = np.abs(Cf - Cp).max() / np.abs(Cp).max()
    dm = np.abs(Mf - ts.reference_).max() / np.abs(ts.reference_).max()
    rho = np.corrcoef(Vf.ravel(), Vp.ravel())[0, 1]
    dv = np.abs(np.sort(np.abs(Vf), 1) - np.sort(np.abs(Vp), 1)).max()
    print("n=%d d=%d T=%d" % (n, d, T))
    print("  cov   max rel diff  %.3e" % dc)
    print("  mean  max rel diff  %.3e" % dm)
    print("  tangent corr        %.10f   (col-order-free max diff %.3e)" % (rho, dv))
    print("  pyriemann %.2fs   fastriem %.2fs   speedup %.1fx"
          % (t1 - t0, t2 - t1, (t1 - t0) / max(t2 - t1, 1e-9)))
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    y = rng.integers(0, 3, n)
    mk = lambda: make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    sp = cross_val_score(mk(), Vp, y, cv=4).mean()
    sf = cross_val_score(mk(), Vf, y, cv=4).mean()
    print("  downstream LDA acc  pyriemann %.4f   fastriem %.4f\n" % (sp, sf))
