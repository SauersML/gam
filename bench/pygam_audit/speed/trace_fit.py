"""Fit once with Rust info logging on; stderr carries the solver trace.

Usage: trace_fit.py FAMILY N DESIGN [K]   (K appended as ', k=K' to every term)
Prints phase timings (perf_counter / process_time) to stdout.
"""
import sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
fam, n, design = sys.argv[1], int(float(sys.argv[2])), sys.argv[3]
k = f", k={sys.argv[4]}" if len(sys.argv) > 4 else ""
rng = np.random.default_rng(0)
if design == "te":
    X = rng.uniform(0, 1, (n, 2)); eta = np.sin(2*np.pi*X[:, 0])*np.cos(2*np.pi*X[:, 1])
else:
    p = int(design[1:]); X = rng.uniform(0, 1, (n, p)); eta = np.zeros(n)
    for j in range(p):
        eta += np.sin(2*np.pi*X[:, j]+j)/np.sqrt(p)
if fam == "gaussian":
    y = eta + rng.normal(0, 0.5, n)
elif fam == "binomial":
    y = (rng.uniform(size=n) < 1/(1+np.exp(-1.5*eta))).astype(float)
else:
    y = rng.poisson(np.exp(0.5+0.7*eta)).astype(float)
names = [f"x{j}" for j in range(X.shape[1])]
data = {nm: X[:, j] for j, nm in enumerate(names)}; data["y"] = y
formula = f"y ~ te(x0, x1{k})" if design == "te" else "y ~ " + " + ".join(f"s({nm}{k})" for nm in names)
import gamfit
import pandas  # noqa: F401  (pre-import so the cold pandas probe is not in the timing)
gamfit._rust.set_log_level("info")
t = time.perf_counter(); c = time.process_time()
m = gamfit.fit(data, formula, family=fam)
print(f"fit wall={time.perf_counter()-t:.3f} cpu={time.process_time()-c:.3f}")
print(m.summary())
