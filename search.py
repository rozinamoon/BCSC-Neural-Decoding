"""
Canonical black-box search, vendored verbatim from type1_learning/core/search.py.

Kept here so the server only needs this repo. Do not diverge: if core/search.py changes,
re-copy. Rationale from the original docstring:

  CMA-ES and NES are the same family -- CMA-ES's rank-mu update is essentially NES's natural
  gradient -- and CMA-ES is the better-tested and more stable, so it is the default TOOL. Tool
  choice is orthogonal to the ORIGIN claim: for origin arguments use selection-only search (no
  covariance / step-size adaptation, which are Lamarckian).

  Diagonal (sep-CMA-ES) by default: full covariance is O(D^2) memory and O(D^3)
  eigendecomposition, and D = n_params(32) = 1248 here.

fitness: callable (population [pop, D]) -> [pop] scores, MAXIMIZED. Rank-based updates, so
step-function survival is fine.
"""
import numpy as np


def _recomb_weights(lam):
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    return mu, w, mueff


class SepCMAES:
    """Separable (diagonal-covariance) CMA-ES. Rank-based; maximizes fitness."""

    def __init__(self, dim, pop=None, sigma0=1.0, mean0=None, seed=0):
        self.D = dim
        self.lam = pop or (4 + int(3 * np.log(dim)))
        self.mu, self.w, self.mueff = _recomb_weights(self.lam)
        D, mueff = float(dim), self.mueff
        self.cs = (mueff + 2) / (D + mueff + 5)
        self.ds = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (D + 1)) - 1) + self.cs
        self.cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
        c1 = 2 / ((D + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
        k = (D + 2) / 3.0
        self.c1, self.cmu = min(1.0, c1 * k), min(1 - min(1.0, c1 * k), cmu * k)
        self.chiN = np.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D * D))
        self.mean = np.zeros(dim) if mean0 is None else np.array(mean0, dtype=float)
        self.sigma = float(sigma0)
        self.c = np.ones(dim)
        self.ps = np.zeros(dim)
        self.pc = np.zeros(dim)
        self.gen = 0
        self.rng = np.random.default_rng(seed)

    def ask(self):
        self.z = self.rng.standard_normal((self.lam, self.D))
        self.y = self.z * np.sqrt(self.c)
        return self.mean + self.sigma * self.y

    def tell(self, fitness):
        idx = np.argsort(-np.asarray(fitness))[: self.mu]
        yw = self.w @ self.y[idx]
        self.mean = self.mean + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs) * self.mueff) * (yw / np.sqrt(self.c))
        self.gen += 1
        hs = np.linalg.norm(self.ps) / np.sqrt(
            1 - (1 - self.cs) ** (2 * self.gen)) / self.chiN < 1.4 + 2 / (self.D + 1)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff) * yw
        rank1 = self.pc ** 2 - self.c
        rankmu = self.w @ (self.y[idx] ** 2 - self.c)
        self.c = np.maximum(self.c + self.c1 * rank1 + self.cmu * rankmu, 1e-12)
        self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(np.clip(self.sigma, 1e-12, 1e6))


def run_cmaes(fitness, dim, gens, pop=None, sigma0=1.0, mean0=None, seed=0, callback=None):
    es = SepCMAES(dim, pop, sigma0, mean0, seed)
    best_x, best_f = None, -np.inf
    for g in range(1, gens + 1):
        X = es.ask()
        f = np.asarray(fitness(X), dtype=float)
        es.tell(f)
        i = int(np.argmax(f))
        if f[i] > best_f:
            best_f, best_x = float(f[i]), X[i].copy()
        if callback is not None:
            callback(g, es, X, f, best_x, best_f)
    return best_x, best_f, es


class SelectionOnly:
    """Mutation + truncation selection, no covariance/step-size adaptation.

    For ORIGIN claims: this is Darwinian (undirected variation + selection), unlike CMA-ES whose
    covariance/step-size adaptation learns a search direction (Lamarckian). Slower, but it is the
    process the origin argument actually asserts.
    """

    def __init__(self, dim, pop=256, sigma=0.08, elite_frac=0.25, seed=0, init_scale=1.0):
        self.rng = np.random.default_rng(seed)
        self.P = self.rng.standard_normal((pop, dim)) * init_scale
        self.sigma, self.n_elite = sigma, max(1, int(pop * elite_frac))

    def ask(self):
        return self.P

    def tell(self, fitness):
        idx = np.argsort(-np.asarray(fitness))[: self.n_elite]
        elite = self.P[idx]
        pop, dim = self.P.shape
        parents = elite[self.rng.integers(0, self.n_elite, pop - self.n_elite)]
        children = parents + self.rng.standard_normal((pop - self.n_elite, dim)) * self.sigma
        self.P = np.concatenate([elite, children])


def run_selection_only(fitness, dim, gens, pop=256, sigma=0.08, elite_frac=0.25,
                       seed=0, init_scale=1.0, callback=None):
    es = SelectionOnly(dim, pop, sigma, elite_frac, seed, init_scale)
    best_x, best_f = None, -np.inf
    for g in range(1, gens + 1):
        X = es.ask()
        f = np.asarray(fitness(X), dtype=float)
        es.tell(f)
        i = int(np.argmax(f))
        if f[i] > best_f:
            best_f, best_x = float(f[i]), X[i].copy()
        if callback is not None:
            callback(g, es, X, f, best_x, best_f)
    return best_x, best_f, es
