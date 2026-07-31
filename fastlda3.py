import numpy as np
from scipy import linalg


def lw_shrinkage(Z: np.ndarray) -> tuple[float, float]:
    n, p = Z.shape
    G = Z @ Z.T
    sq = np.einsum("ij,ij->i", Z, Z)
    mu = sq.sum() / (n * p)
    delta_ = (G ** 2).sum() / n ** 2
    beta_ = (1.0 / (p * n)) * ((sq ** 2).sum() / n - delta_)
    delta = (delta_ - 2.0 * mu * sq.sum() / n + p * mu ** 2) / p
    beta = min(beta_, delta)
    sh = 0.0 if beta == 0 else beta / delta
    return float(min(max(sh, 0.0), 1.0)), float(mu)


def cov_auto(Xg: np.ndarray) -> np.ndarray:
    sd = Xg.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (Xg - Xg.mean(0)) / sd
    sh, mu = lw_shrinkage(Z)
    S = (Z.T @ Z) * ((1.0 - sh) / len(Z))
    S[np.diag_indices_from(S)] += sh * mu
    S *= sd[:, None]
    S *= sd[None, :]
    return S


def fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cls = np.unique(y)
    M = np.stack([X[y == c].mean(0) for c in cls], 0)
    pri = np.array([(y == c).mean() for c in cls])
    S = np.zeros((X.shape[1], X.shape[1]))
    for i, c in enumerate(cls):
        S += pri[i] * cov_auto(X[y == c])
    coef = linalg.solve(S, M.T, assume_a="pos").T
    inter = -0.5 * np.einsum("ij,ij->i", M, coef) + np.log(pri)
    return coef, inter, cls


def predict(X: np.ndarray, coef: np.ndarray, inter: np.ndarray,
            cls: np.ndarray) -> np.ndarray:
    return cls[(X @ coef.T + inter).argmax(1)]


if __name__ == "__main__":
    import time
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    rng = np.random.default_rng(0)
    n, C = 176, 11
    y = np.repeat(np.arange(C), n // C)
    yt = np.repeat(np.arange(C), 4)
    print("  p      agree   coef_rel   dec_corr         sklearn     fast    speedup")
    for p, s in [(300, 0.05), (759, 0.04), (1518, 0.03)]:
        sig = rng.normal(size=(C, p)) * s
        X = sig[y] + rng.normal(size=(n, p))
        Xt = sig[yt] + rng.normal(size=(len(yt), p))
        t0 = time.time()
        sk = LDA(solver="lsqr", shrinkage="auto").fit(X, y)
        ps = sk.predict(Xt)
        t1 = time.time()
        co, it, cl = fit(X, y)
        pc = predict(Xt, co, it, cl)
        t2 = time.time()
        d1, d2 = sk.decision_function(Xt), Xt @ co.T + it
        print("  %-6d %.4f  %.3e  %.12f   %7.2fs %7.3fs   %5.0fx"
              % (p, (ps == pc).mean(),
                 np.abs(co - sk.coef_).max() / np.abs(sk.coef_).max(),
                 np.corrcoef(d1.ravel(), d2.ravel())[0, 1],
                 t1 - t0, t2 - t1, (t1 - t0) / max(t2 - t1, 1e-9)))
    print()
    for p in [2970, 5940]:
        X = rng.normal(size=(n, p))
        t0 = time.time()
        fit(X, y)
        print("  p=%-5d fast %.3fs   (sklearn extrapolated %.0fs)"
              % (p, time.time() - t0, 14.9 * (p / 1518) ** 3))
