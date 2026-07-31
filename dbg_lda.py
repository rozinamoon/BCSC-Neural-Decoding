import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import fastlda as fl

rng = np.random.default_rng(0)
n, C, p = 176, 11, 400
y = np.repeat(np.arange(C), n // C)
X = rng.normal(size=(n, p))

sk = LDA(solver="lsqr", shrinkage="auto").fit(X, y)
S_sk = sk.covariance_

cls = np.unique(y)
M = np.stack([X[y == c].mean(0) for c in cls], 0)
pri = np.array([(y == c).mean() for c in cls])
A = 0.0
rows = []
for i, c in enumerate(cls):
    Z = X[y == c] - M[i]
    sh, mu = fl._lw_shrinkage(Z)
    A += pri[i] * sh * mu
    rows.append(np.sqrt(pri[i] * (1 - sh) / len(Z)) * Z)
B = np.concatenate(rows, 0)
S_me = A * np.eye(p) + B.T @ B

print("covariance   max rel diff  %.3e" % (np.abs(S_me - S_sk).max() / np.abs(S_sk).max()))
print("means        max rel diff  %.3e" % (np.abs(M - sk.means_).max() / np.abs(sk.means_).max()))
print("cond(Sigma)  %.3e" % np.linalg.cond(S_sk))

c_lstsq = linalg.lstsq(S_sk, M.T)[0].T
c_solve = np.linalg.solve(S_sk, M.T).T
co, it, cl = fl.fit_dual(X, y)
f = lambda a, b: np.abs(a - b).max() / np.abs(b).max()
print("\ncoef  sklearn(lstsq) vs np.solve      %.3e" % f(c_lstsq, c_solve))
print("coef  sklearn(lstsq) vs dual          %.3e" % f(sk.coef_, co))
print("coef  np.solve       vs dual          %.3e" % f(c_solve, co))
print("\nresidual ||S c - M||  lstsq %.3e   solve %.3e   dual %.3e"
      % (np.abs(S_sk @ c_lstsq.T - M.T).max(),
         np.abs(S_sk @ c_solve.T - M.T).max(),
         np.abs(S_sk @ co.T - M.T).max()))
sv = linalg.svdvals(S_sk)
print("\nsingular values: max %.4e  min %.4e  ratio %.3e" % (sv[0], sv[-1], sv[-1] / sv[0]))
print("lstsq default cond cutoff would drop %d of %d"
      % (int((sv < np.finfo(float).eps * sv[0] * max(S_sk.shape)).sum()), len(sv)))
