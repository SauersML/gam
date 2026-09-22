"""Time binomial fits on pyGAM's `default` dataset (n=10000) by formula; report CPU and wall."""
import os, sys, time, resource, warnings
import numpy as np
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
import pygam.datasets.load_datasets as _L
from pg_helpers import dataset_dir  # noqa: E402
_L.PATH = dataset_dir()
from pygam import datasets as _ds
import gamfit
X, y = _ds.default(return_X_y=True)
n = int(sys.argv[2]) if len(sys.argv) > 2 else len(y)
d = {"student": np.asarray([f"s{int(v)}" for v in X[:n, 0]]), "balance": X[:n, 1].astype(float),
     "income": X[:n, 2].astype(float), "y": np.asarray(y[:n], float)}
f = sys.argv[1]
t, c = time.time(), time.process_time()
m = gamfit.fit(d, f, family="binomial")
conv = m.summary().convergence
print(n, f, f"wall {time.time()-t:.1f}s cpu {time.process_time()-c:.1f}s", "certified", conv.get("certified"), "edf", m.summary().edf_total, flush=True)
