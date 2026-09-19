"""gamfit expectile: accuracy at several tau, crossing across tau, CI coverage; vs pyGAM ExpectileGAM (gridsearch)."""
import numpy as np, warnings, time, sys
from scipy import optimize, stats
import gamfit
from pygam import ExpectileGAM, s
warnings.filterwarnings("ignore")

def e_std(tau):
    g = lambda e: tau * (stats.norm.pdf(e) - e * (1 - stats.norm.cdf(e))) - (1 - tau) * (e * stats.norm.cdf(e) + stats.norm.pdf(e))
    return optimize.brentq(g, -10, 10)

taus = [0.05, 0.5, 0.95]
G = np.linspace(0.02, 0.98, 100)
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
res = {("gamfit", t): [] for t in taus} | {("pygam_gs", t): [] for t in taus}
cross = {"gamfit": 0, "pygam_gs": 0}
for r in range(reps):
    rng = np.random.default_rng(500 + r); n = 800
    x = rng.uniform(0, 1, n); sd = 0.2 + 0.8 * x
    y = np.sin(2 * np.pi * x) + sd * rng.normal(size=n)
    preds = {"gamfit": [], "pygam_gs": []}
    for t in taus:
        truth = np.sin(2 * np.pi * G) + (0.2 + 0.8 * G) * e_std(t)
        t0 = time.perf_counter()
        m = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="expectile", expectile_tau=t)
        tt = time.perf_counter() - t0
        o = m.predict({"x": G}, interval=0.95, return_type="dict")
        mu = np.asarray(o["posterior_mean"]); lo = np.asarray(o["posterior_mean_lower"]); hi = np.asarray(o["posterior_mean_upper"])
        res[("gamfit", t)].append((np.sqrt(np.mean((mu - truth) ** 2)), np.mean((lo <= truth) & (truth <= hi)), tt))
        preds["gamfit"].append(mu)
        t0 = time.perf_counter()
        pm = ExpectileGAM(s(0), expectile=t).gridsearch(x[:, None], y, progress=False)
        tt = time.perf_counter() - t0
        mu = pm.predict(G[:, None]); ci = pm.confidence_intervals(G[:, None], width=0.95)
        res[("pygam_gs", t)].append((np.sqrt(np.mean((mu - truth) ** 2)), np.mean((ci[:, 0] <= truth) & (truth <= ci[:, 1])), tt))
        preds["pygam_gs"].append(mu)
    for k, p in preds.items():
        p = np.array(p)
        cross[k] += int(np.sum((np.diff(p, axis=0) < 0).any(axis=0)))
for (k, t), v in res.items():
    a = np.array(v)
    print(f"{k:9s} tau={t:.2f} RMSE {a[:,0].mean():.4f} cover95 {a[:,1].mean():.3f} time {a[:,2].mean():.2f}s")
print("grid points (of", 100 * reps, ") where fitted expectiles cross:", cross)
