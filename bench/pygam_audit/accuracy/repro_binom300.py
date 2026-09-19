"""Repro: binom_add4_n300 fold 0 -> gamfit raises 'did not certify a stationary optimum'."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np, gamfit
from sklearn.model_selection import StratifiedKFold

def gamsim(X):
    f0 = 2 * np.sin(np.pi * X[:, 0]); f1 = np.exp(2 * X[:, 1])
    f2 = 0.2 * X[:, 2] ** 11 * (10 * (1 - X[:, 2])) ** 6 + 10 * (10 * X[:, 2]) ** 3 * (1 - X[:, 2]) ** 10
    return f0 + f1 + f2
n = 300
rng = np.random.default_rng(n + 4)
X = rng.uniform(0, 1, (n, 4))
mu = 1 / (1 + np.exp(-(gamsim(X) - 7.5) / 1.5))
y = (rng.uniform(size=n) < mu).astype(float)
tr, te = next(iter(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)))
d = {f"x{j}": X[tr, j] for j in range(4)}; d["y"] = y[tr]
np.savez("binom300_fold0.npz", **d)
try:
    m = gamfit.fit(d, "y ~ s(x0) + s(x1) + s(x2) + s(x3)", family="binomial")
    print("OK edf", m.summary().edf_total)
except Exception as e:
    print(type(e).__name__, str(e)[:500])
