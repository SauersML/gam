"""Probe which formula features gamfit accepts (fit succeeds) vs rejects."""
import sys
from common import *

rng = np.random.default_rng(0)
n = 400
x = rng.uniform(0, 1, n)
z = rng.uniform(0, 1, n)
w = rng.uniform(0.5, 2, n)
g = rng.choice(["a", "b", "c"], n)
gi = rng.integers(0, 4, n).astype(float)
t = rng.uniform(0, 24, n)
y = np.sin(3 * x) + z ** 2 + (g == "b") * 0.5 + 0.3 * np.cos(2 * np.pi * t / 24) + rng.normal(0, 0.3, n)
data = dict(x=x, z=z, w=w, g=g, gi=gi, t=t, y=y)

probes = [
    # basics
    "y ~ s(x) + z",
    "y ~ s(x) + linear(z)",
    "y ~ s(x) + factor(g)",
    "y ~ s(x) + g",
    "y ~ s(x) + factor(gi)",
    "y ~ s(x) - 1",
    "y ~ 0 + s(x)",
    "y ~ -1 + s(x)",
    # n_splines / order
    "y ~ s(x, k=20)",
    "y ~ s(x, k=20, degree=1)",
    "y ~ s(x, k=20, degree=0)",
    "y ~ s(x, k=20, degree=5)",
    # penalties
    "y ~ s(x, penalty_order=1)",
    "y ~ s(x, penalty_order=3)",
    "y ~ s(x, penalty_order=[1,2])",
    "y ~ s(x, penalty_order=all)",
    "y ~ s(x, double_penalty=false)",
    "y ~ s(x, fx=true)",
    "y ~ s(x, penalty=none)",
    "y ~ s(x, sp=10)",
    "y ~ s(x, lambda=10)",
    "y ~ s(x, lam=10)",
    # constraints
    "y ~ s(x, shape=monotone_increasing)",
    "y ~ s(x, shape=convex)",
    "y ~ s(x, shape=[monotone_increasing, convex])",
    "y ~ s(x, shape='monotone_increasing,concave')",
    "y ~ s(x, shape=monotone_increasing, degree=1)",
    "y ~ s(x, shape=monotone_increasing, degree=2)",
    "y ~ s(x, shape=monotone_increasing, knot_placement=uniform)",
    "y ~ s(x, shape=monotone_increasing, bc=clamped)",
    "y ~ s(x, shape=monotone_increasing, identifiability=none)",
    "y ~ s(x, shape=monotone_increasing, type=duchon)",
    "y ~ duchon(x, shape=monotone_increasing)",
    "y ~ tps(x, shape=monotone_increasing)",
    "y ~ matern(x, shape=monotone_increasing)",
    "y ~ s(x, bs=cr, shape=monotone_increasing)",
    "y ~ s(t, period=24, shape=convex)",
    # periodic
    "y ~ cyclic(t, period=24)",
    "y ~ s(t, period=24)",
    "y ~ cp(t, period_start=0, period_end=24)",
    # by
    "y ~ s(x, by=z)",
    "y ~ s(x, by=g) + g",
    "y ~ s(x, by=g, shape=monotone_increasing) + g",
    "y ~ s(x, by=z, shape=monotone_increasing)",
    "y ~ te(x, z, by=w)",
    "y ~ te(x, z, by=g) + g",
    # edge knots / domain
    "y ~ s(x, range=[-1, 2])",
    "y ~ s(x, domain=[-1, 2])",
    "y ~ s(x, boundary_knots=[-1, 2])",
    "y ~ s(x, knot_placement=uniform)",
    "y ~ s(x, knot_placement=quantile)",
    # tensor
    "y ~ te(x, z)",
    "y ~ te(x, z, k=[6, 8])",
    "y ~ te(x, z, degree=[1, 3], k=6)",
    "y ~ te(x, z, penalty_order=[1, 2])",
    "y ~ te(x, z, bs=[ps, cr])",
    "y ~ te(x, t, periods=[None, 24])",
    "y ~ te(x, z, shape=monotone_increasing)",
    "y ~ te(x, z, shape=[monotone_increasing, none])",
    "y ~ ti(x, z) + s(x) + s(z)",
    "y ~ te(x, g)",
    "y ~ te(x, z, w)",
    # categorical smooth
    "y ~ s(g, bs=re)",
    "y ~ s(g)",
    "y ~ s(gi, dtype=categorical)",
]

res = []
for p in probes:
    try:
        m = gamfit.fit(data, p)
        pred = gpred(m, data)
        status = "OK"
        msg = f"rmse_in={np.sqrt(np.mean((pred - y) ** 2)):.3f}"
    except Exception as e:
        status = "ERR"
        msg = f"{type(e).__name__}: {str(e).splitlines()[0][:220]}"
    print(f"{status:3s} | {p:55s} | {msg}", flush=True)
