"""Paired per-seed comparison, 1-D heteroscedastic design: gamfit expectile vs pyGAM ExpectileGAM gridsearch at tau 0.5, 0.95."""
import numpy as np, warnings, sys
from scipy import optimize, stats
import gamfit
from pygam import ExpectileGAM, s
warnings.filterwarnings("ignore")
def e_std(tau):
    g = lambda e: tau * (stats.norm.pdf(e) - e * (1 - stats.norm.cdf(e))) - (1 - tau) * (e * stats.norm.cdf(e) + stats.norm.pdf(e))
    return optimize.brentq(g, -10, 10)
G = np.linspace(0.02, 0.98, 100); reps = int(sys.argv[1])
for t in (0.5, 0.95):
    d = []
    for r in range(reps):
        rng = np.random.default_rng(700 + r); n = 800
        x = rng.uniform(0, 1, n); y = np.sin(2 * np.pi * x) + (0.2 + 0.8 * x) * rng.normal(size=n)
        truth = np.sin(2 * np.pi * G) + (0.2 + 0.8 * G) * e_std(t)
        m = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="expectile", expectile_tau=t)
        o = m.predict({"x": G}, interval=0.95, return_type="dict")
        a = np.sqrt(np.mean((np.asarray(o["posterior_mean"]) - truth) ** 2))
        ca = np.mean((np.asarray(o["posterior_mean_lower"]) <= truth) & (truth <= np.asarray(o["posterior_mean_upper"])))
        pm = ExpectileGAM(s(0), expectile=t).gridsearch(x[:, None], y, progress=False)
        b = np.sqrt(np.mean((pm.predict(G[:, None]) - truth) ** 2))
        d.append((a, b, ca))
        print(t, r, round(a, 4), round(b, 4), round(ca, 3), flush=True)
    d = np.array(d)
    print(f"tau={t}: gamfit RMSE {d[:,0].mean():.4f}  pyGAM-gs {d[:,1].mean():.4f}  gamfit wins {int((d[:,0]<d[:,1]).sum())}/{reps}  gamfit cover95 {d[:,2].mean():.3f}", flush=True)
