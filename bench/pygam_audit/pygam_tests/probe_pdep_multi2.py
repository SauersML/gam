"""Additive identity eta == intercept + sum_j pdep_j(x_ij) on a Poisson multi-term model (chicago subset)."""
import inspect
import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pg_helpers import eta_of, intercept, pdep  # noqa: E402
import pygam.datasets.load_datasets as _L  # noqa: E402

_L.PATH = os.path.join(HERE, "data")
from pygam import datasets as _ds  # noqa: E402
import gamfit  # noqa: E402

X, y = _ds.chicago(return_X_y=True)
keep = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
X, y = X[keep][:1500], np.asarray(y, float)[keep][:1500]
d = {"time": X[:, 0], "tmpd": X[:, 1], "pm10": X[:, 2], "o3": X[:, 3], "y": y}
m = gamfit.fit(d, "y ~ s(time) + s(tmpd)", family="poisson")
eta = eta_of(m, d)
tot = intercept(m, d)
tot = tot + pdep(m, "s(time)", d, grid=d["time"])["predicted"]
tot = tot + pdep(m, "s(tmpd)", d, grid=d["tmpd"])["predicted"]
print("max |eta - (b0 + sum pdep)| =", float(np.max(np.abs(eta - tot))), " max|eta| =", float(np.max(np.abs(eta))))
