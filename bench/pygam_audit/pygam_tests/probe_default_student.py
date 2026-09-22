"""Is the student effect in pyGAM's `default` model shrunk away by factor() being a penalized random effect (F1)?"""
import os, warnings
import numpy as np
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
import pygam.datasets.load_datasets as _L
from pg_helpers import dataset_dir  # noqa: E402
_L.PATH = dataset_dir()
from pygam import datasets as _ds
import gamfit
X, y = _ds.default(return_X_y=True)
n = 2000
z = lambda v: (v - v.mean()) / v.std()
d = {"student": np.asarray([f"s{int(v)}" for v in X[:n, 0]]), "stud01": X[:n, 0].astype(float),
     "balance": z(X[:n, 1].astype(float)), "income": z(X[:n, 2].astype(float)), "y": np.asarray(y[:n], float)}
g = {k: np.array(v[:2]) for k, v in d.items()}
g["balance"] = np.zeros(2); g["income"] = np.zeros(2)
g["student"] = np.array(["s0", "s1"]); g["stud01"] = np.array([0.0, 1.0])
for f in ["y ~ factor(student) + s(balance) + s(income)", "y ~ linear(stud01) + s(balance) + s(income)"]:
    m = gamfit.fit(d, f, family="binomial")
    eta = np.log(m.predict(g) / (1 - m.predict(g)))
    s = m.summary()
    print(f, "| student log-odds effect (s1 - s0):", float(eta[1] - eta[0]), "| lambdas", s.lambdas, "| edf", s.edf_total, flush=True)
