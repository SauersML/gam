"""Per-call predict overhead: gamfit vs pyGAM at small n (fixed cost dominates).

Usage: pred_overhead.py FAMILY N
Fits s(x) once, then times 20 predict calls on 1000 fresh rows and profiles one.
"""
import cProfile
import pstats
import sys
import time

import numpy as np

fam, n = sys.argv[1], int(float(sys.argv[2]))
rng = np.random.default_rng(0)
x = rng.uniform(0, 1, n)
eta = np.sin(2 * np.pi * x)
if fam == "gaussian":
    y = eta + rng.normal(0, 0.5, n)
elif fam == "binomial":
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * eta))).astype(float)
else:
    y = rng.poisson(np.exp(0.5 + 0.7 * eta)).astype(float)
import gamfit
import pandas  # noqa: F401

m = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family=fam)
xp = {"x": rng.uniform(0, 1, 1000)}
m.predict(xp)
ts = []
for _ in range(20):
    c = time.process_time()
    m.predict(xp)
    ts.append(time.process_time() - c)
print(f"gamfit {fam} predict(1000 rows) cpu median={np.median(ts)*1e3:.2f} ms min={min(ts)*1e3:.2f} ms")
pr = cProfile.Profile()
pr.enable()
for _ in range(5):
    m.predict(xp)
pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(14)
