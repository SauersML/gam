"""Scope the 'smoothing cubature has no positive-width proposal' IntegrationError."""
import os
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
import pygam.datasets.load_datasets as _L  # noqa: E402

from pg_helpers import dataset_dir  # noqa: E402
_L.PATH = dataset_dir()
from pygam import datasets as _ds  # noqa: E402
import gamfit  # noqa: E402

X, y = _ds.chicago(return_X_y=True)
y = np.asarray(y, float)
print("chicago n", len(y), "nonfinite X rows", int(np.sum(~np.all(np.isfinite(X), axis=1))), flush=True)
keep = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
X, y = X[keep], y[keep]
cases = sys.argv[1:] or ["1500|y ~ s(time) + s(tmpd) + te(pm10, o3)"]
for case in cases:
    n, f = case.split("|")
    n = int(n)
    d = {"time": X[:n, 0], "tmpd": X[:n, 1], "pm10": X[:n, 2], "o3": X[:n, 3], "y": y[:n]}
    t = time.time()
    try:
        m = gamfit.fit(d, f, family="poisson")
        c = m.summary().convergence
        print(n, f, "OK", f"{time.time()-t:.0f}s", "certified", c.get("certified"), flush=True)
    except Exception as e:
        print(n, f, "ERR", type(e).__name__, str(e)[:150], f"{time.time()-t:.0f}s", flush=True)
