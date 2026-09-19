"""Re-run exactly one worker.py config (same data generator and seed) with Rust info logging.

Usage: trace_worker_seed.py FAMILY N DESIGN SEED   (stderr = solver trace)
"""
import sys
import time

import numpy as np

fam, n, design, seed = sys.argv[1], int(float(sys.argv[2])), sys.argv[3], int(sys.argv[4])
rng = np.random.default_rng(seed)
if design == "te":
    X = rng.uniform(0, 1, (n, 2))
    eta = np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])
else:
    p = int(design[1:])
    X = rng.uniform(0, 1, (n, p))
    eta = np.zeros(n)
    for j in range(p):
        eta += np.sin(2 * np.pi * X[:, j] + j) / np.sqrt(p)
if fam == "gaussian":
    y = eta + rng.normal(0, 0.5, n)
elif fam == "binomial":
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * eta))).astype(float)
else:
    y = rng.poisson(np.exp(0.5 + 0.7 * eta)).astype(float)
names = [f"x{j}" for j in range(X.shape[1])]
data = {nm: X[:, j] for j, nm in enumerate(names)}
data["y"] = y
formula = "y ~ te(x0, x1)" if design == "te" else "y ~ " + " + ".join(f"s({nm})" for nm in names)
import gamfit

gamfit._rust.set_log_level("info")
t = time.perf_counter(); c = time.process_time()
m = gamfit.fit(data, formula, family=fam)
print(f"fit wall={time.perf_counter()-t:.3f} cpu={time.process_time()-c:.3f}")
print(m.summary())
