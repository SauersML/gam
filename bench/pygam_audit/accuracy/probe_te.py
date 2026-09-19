"""bump2d n=4000: default te() vs larger margins; truth MSE on held-out fold 0."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[v] = "1"
import warnings, numpy as np, gamfit
from sklearn.model_selection import KFold
from pygam import LinearGAM, te
n = 4000
rng = np.random.default_rng(n + 1)
X = rng.uniform(0, 1, (n, 2))
mu = 3*np.exp(-((X[:,0]-.3)**2+(X[:,1]-.6)**2)/.05) + 2*np.exp(-((X[:,0]-.75)**2+(X[:,1]-.25)**2)/.02)
y = mu + rng.normal(0, .3, n)
tr, tst = next(iter(KFold(5, shuffle=True, random_state=0).split(X)))
d = {"x0": X[tr,0], "x1": X[tr,1], "y": y[tr]}
dt = {"x0": X[tst,0], "x1": X[tst,1]}
for f in ["y ~ te(x0, x1)", "y ~ te(x0, x1, k=8)", "y ~ te(x0, x1, k=10)", "y ~ te(x0, x1, k=14)"]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = gamfit.fit(d, f)
    p = m.predict(dt)
    ncol = m.design_matrix(dt)
    try: ncol = ncol.shape
    except Exception: ncol = type(ncol).__name__
    print(f"{f:28s} truth_mse={np.mean((p-mu[tst])**2):.5f} edf={m.summary().edf_total:.1f} design={ncol} warn={[str(x.message)[:140] for x in w]}")
g = LinearGAM(te(0, 1)).gridsearch(X[tr], y[tr], progress=False)
print("pygam grid te(0,1) truth_mse", np.mean((g.predict(X[tst]) - mu[tst])**2), "edof", g.statistics_["edof"])
