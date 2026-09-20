"""pois_lowcount_n500: gamfit default vs k=20 vs pyGAM, truth MSE on each held-out fold."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[v] = "1"
import warnings, numpy as np, gamfit
from sklearn.model_selection import KFold
from pygam import PoissonGAM, s
warnings.simplefilter("ignore")
n = 500
rng = np.random.default_rng(n + 9)
x = rng.uniform(0, 1, n)
mu = np.exp(-1.5 + 1.5 * np.sin(2 * np.pi * 2 * x))
y = rng.poisson(mu).astype(float)
res = {k: [] for k in ("default", "k20", "pygam", "pygam_grid")}
for tr, te in KFold(5, shuffle=True, random_state=0).split(x):
    for key, f in (("default", "y ~ s(x)"), ("k20", "y ~ s(x, k=20)")):
        m = gamfit.fit({"x": x[tr], "y": y[tr]}, f, family="poisson")
        res[key].append((np.mean((m.predict({"x": x[te]}) - mu[te]) ** 2), m.summary().edf_total))
    g = PoissonGAM(s(0)).fit(x[tr, None], y[tr])
    res["pygam"].append((np.mean((g.predict_mu(x[te, None]) - mu[te]) ** 2), g.statistics_["edof"]))
    g = PoissonGAM(s(0)).gridsearch(x[tr, None], y[tr], progress=False)
    res["pygam_grid"].append((np.mean((g.predict_mu(x[te, None]) - mu[te]) ** 2), g.statistics_["edof"]))
for k, v in res.items():
    v = np.array(v)
    print(f"{k:10s} truth_mse per fold {np.round(v[:,0],5)} mean {v[:,0].mean():.5f} edf {np.round(v[:,1],1)}")
