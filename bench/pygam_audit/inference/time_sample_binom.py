import os, sys, time, warnings
os.environ["RAYON_NUM_THREADS"]="1"
import numpy as np, gamfit
from mc import CELLS, N_TEST, f1, f3, inv_link, draw_y
warnings.simplefilter("ignore")
fam, n, b0, a1, a3, sigma = CELLS["binom"]
Xt = np.random.default_rng(12345).uniform(0.02, 0.98, (N_TEST, 3))
for rep in [1,2,3]:
    rng = np.random.default_rng(1000 + rep)
    X = rng.uniform(0, 1, (n, 3))
    y = draw_y(fam, inv_link(fam, b0 + f1(X[:, 0], a1) + f3(X[:, 2], a3)), sigma, rng)
    mu_t = inv_link(fam, b0 + f1(Xt[:, 0], a1) + f3(Xt[:, 2], a3))
    d = dict(x1=X[:, 0], x2=X[:, 1], x3=X[:, 2], y=y)
    dt = dict(x1=Xt[:, 0], x2=Xt[:, 1], x3=Xt[:, 2])
    t0=time.time()
    try:
        m = gamfit.fit(d, "y ~ s(x1) + s(x2) + s(x3)", family=fam)
    except Exception as e:
        print(rep, "fit error", time.time()-t0, repr(e)[:150], flush=True); continue
    print(rep, "fit", round(time.time()-t0,1), flush=True)
    for S in [20, 100, 500]:
        t0=time.time()
        ps = m.sample(d, samples=S, seed=rep)
        r = ps.predict(dt, level=0.95)
        lo, hi = np.asarray(r["posterior_mean_lower"]), np.asarray(r["posterior_mean_upper"])
        print(rep, "samples", S, "time", round(time.time()-t0,1), ps.method, ps.covariance_source, "cover", np.mean((mu_t>=lo)&(mu_t<=hi)), "width", np.mean(hi-lo), flush=True)
