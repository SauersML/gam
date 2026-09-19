"""Repro: nearsep_n200 folds 0 and 2 -> gamfit IntegrationError (outer not certified)."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np, gamfit
from sklearn.model_selection import StratifiedKFold
n = 200
rng = np.random.default_rng(n + 6)
x = rng.uniform(0, 1, n); z = rng.uniform(0, 1, n)
mu = 1 / (1 + np.exp(-(25 * (x - 0.5) + np.sin(2 * np.pi * z))))
y = (rng.uniform(size=n) < mu).astype(float)
for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(x, y)):
    d = {"x": x[tr], "z": z[tr], "y": y[tr]}
    try:
        m = gamfit.fit(d, "y ~ s(x) + s(z)", family="binomial")
        print(k, "OK edf", round(m.summary().edf_total, 2))
    except Exception as e:
        print(k, type(e).__name__, str(e)[:260])
