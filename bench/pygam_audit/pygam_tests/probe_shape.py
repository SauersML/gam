"""Fast probe of the shape-constraint translations (prints per-check results with timing)."""
import sys
import time

import numpy as np

import gamfit
import pandas as pd

A = "/tmp/claude-0/-home-user-gam/02aeec89-32a7-52a0-8d71-90f383516996/scratchpad/audit/pygam_tests"
import pygam.datasets.load_datasets as _L
_L.PATH = A + "/data"
from pygam import datasets as _ds
X, y = _ds.hepatitis(return_X_y=True)
d = {"x": X[:, 0].astype(float), "y": np.asarray(y, float)}
SH = {"monotone_increasing": (1, 1), "monotone_decreasing": (1, -1), "convex": (2, 1), "concave": (2, -1)}
g = np.linspace(d["x"].min(), d["x"].max(), 200)
shapes = sys.argv[1:] or list(SH)
nsamp = 40


def worst(v, o, s):
    v = np.asarray(v, float)
    return float(np.min(s * np.diff(v, n=o, axis=-1))) / max(1.0, float(np.max(np.abs(v))))


for sh in shapes:
    o, s = SH[sh]
    t = time.time()
    m = gamfit.fit(d, f"y ~ s(x, shape={sh})")
    r = m.predict({"x": g}, interval=0.95)
    name = [b.name for b in m.term_blocks if b.kind != "intercept"][0]
    p = m.partial_dependence(name, d, grid=g) if "data" in m.partial_dependence.__code__.co_varnames else m.partial_dependence(name, grid=g)
    print(sh, f"fit {time.time()-t:.1f}s",
          "plugin", f"{worst(r['mean_plugin'], o, s):.2e}",
          "post_mean", f"{worst(r['posterior_mean'], o, s):.2e}",
          "lower", f"{worst(r['posterior_mean_lower'], o, s):.2e}",
          "upper", f"{worst(r['posterior_mean_upper'], o, s):.2e}",
          "pdep", f"{worst(p['predicted'], o, s):.2e}",
          "cov_src", r.get("covariance_source"), flush=True)
    t = time.time()
    post = m.sample(d, samples=nsamp, seed=0)
    eta = np.asarray(post.predict_draws({"x": np.linspace(g[0], g[-1], 60)}).eta, float)
    bad = np.mean([worst(e, o, s) < -1e-8 for e in eta])
    print(sh, f"sample {time.time()-t:.1f}s draws", eta.shape, "frac_violating", bad, flush=True)
