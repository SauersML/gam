#!/usr/bin/env python3
"""ACC-6 evidence on independent replicates, with gamfit itself.

CV folds share training data, so a paired t over folds overstates
significance. This script draws R independent datasets from each
non-Gaussian bench_accuracy generator (same truth, same n, fresh seeds),
fits gamfit once per dataset and scores truth-MSE on a fresh test sample of
2000 covariate draws for three predictions built from the fit's exported
affine design (as in rho_marginal_bench.py):

  plugin       g^{-1}(eta_hat)
  conditional  E[g^{-1}(eta)], eta ~ N(eta_hat, x' V_beta x)   (shipped)
  marginal     E[g^{-1}(eta)], eta ~ N(eta_hat, x' V_p x)      (first-order rho-marginal)

Replicates are independent, so the paired t over them is a valid test.
Self-contained: the generators are copied from bench_accuracy.py.

Usage: python rho_marginal_replicates.py GENERATOR R SEED0
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from pathlib import Path

import numpy as np

import gamfit

HERE = Path(__file__).resolve().parent
N_TEST = 2000

# Same expectations as rho_marginal_bench.py (see its docstring for the
# trapezoid-rule accuracy argument).
Z_NODES = np.linspace(-12.0, 12.0, 8001)
Z_WEIGHTS = np.exp(-0.5 * Z_NODES**2) / np.sqrt(2.0 * np.pi) * (Z_NODES[1] - Z_NODES[0])
Z_WEIGHTS[[0, -1]] *= 0.5


def expect_mean(family, eta, s2):
    if family in ("poisson", "gamma"):
        return np.exp(eta + 0.5 * s2)
    z = eta[:, None] + np.sqrt(np.maximum(s2, 0.0))[:, None] * Z_NODES[None, :]
    return (0.5 * (1.0 + np.tanh(0.5 * z))) @ Z_WEIGHTS


def inv_link(family, eta):
    return np.exp(eta) if family in ("poisson", "gamma") else 1.0 / (1.0 + np.exp(-eta))


def gamsim(X):
    f0 = 2 * np.sin(np.pi * X[:, 0])
    f1 = np.exp(2 * X[:, 1])
    f2 = 0.2 * X[:, 2] ** 11 * (10 * (1 - X[:, 2])) ** 6 + 10 * (10 * X[:, 2]) ** 3 * (1 - X[:, 2]) ** 10
    return f0 + f1 + f2


def logistic(eta):
    return 1 / (1 + np.exp(-eta))


# name -> (family, formula, n, number of covariates, truth mu(X))
GENERATORS = {
    "binom_add4_n300": ("binomial", "y ~ s(x0) + s(x1) + s(x2) + s(x3)", 300, 4,
                        lambda X: logistic((gamsim(X) - 7.5) / 1.5)),
    "binom_add4_n1000": ("binomial", "y ~ s(x0) + s(x1) + s(x2) + s(x3)", 1000, 4,
                         lambda X: logistic((gamsim(X) - 7.5) / 1.5)),
    "binom_sin2_n500": ("binomial", "y ~ s(x0)", 500, 1,
                        lambda X: logistic(2 * np.sin(2 * np.pi * 2 * X[:, 0]))),
    "pois_add2_n300": ("poisson", "y ~ s(x0) + s(x1)", 300, 2,
                       lambda X: np.exp(1 + np.sin(2 * np.pi * 1.5 * X[:, 0]) + 0.8 * np.cos(2 * np.pi * X[:, 1]))),
    "pois_lowcount_n500": ("poisson", "y ~ s(x0)", 500, 1,
                           lambda X: np.exp(-1.5 + 1.5 * np.sin(2 * np.pi * 2 * X[:, 0]))),
    # annual counts on a flat-then-falling rate: the coal-mining shape
    "coal_like_n150": ("poisson", "y ~ s(x0)", 150, 1,
                       lambda X: np.exp(1.0 - 2.2 / (1 + np.exp(-12 * (X[:, 0] - 0.35))))),
    "gamma_add2_n300": ("gamma", "y ~ s(x0) + s(x1)", 300, 2,
                        lambda X: np.exp(0.5 + np.sin(2 * np.pi * X[:, 0]) + X[:, 1] ** 2)),
}


def draw(family, mu, rng):
    if family == "binomial":
        return (rng.uniform(size=len(mu)) < mu).astype(float)
    if family == "poisson":
        return rng.poisson(mu).astype(float)
    return rng.gamma(2.0, mu / 2.0)


def one(name, seed):
    family, formula, n, d, truth = GENERATORS[name]
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    Xt = rng.uniform(0, 1, (N_TEST, d))
    data = {f"x{j}": X[:, j] for j in range(d)}
    data["y"] = draw(family, truth(X), rng)
    new = {f"x{j}": Xt[:, j] for j in range(d)}
    mt = truth(Xt)
    m = gamfit.fit(data, formula, family=family)
    a = m.design_matrix(new)
    G = np.asarray(a.eta_gradient)
    eta = np.asarray(a.offset) + np.asarray(a.matrix) @ np.asarray(a.coefficients)
    s2c = np.einsum("ij,jk,ik->i", G, np.asarray(a.covariance_conditional), G)
    preds = {"plugin": inv_link(family, eta), "conditional": expect_mean(family, eta, s2c)}
    if a.covariance_smoothing_corrected is not None:
        s2p = np.einsum("ij,jk,ik->i", G, np.asarray(a.covariance_smoothing_corrected), G)
        preds["marginal"] = expect_mean(family, eta, s2p)
    return {k: float(np.mean((v - mt) ** 2)) for k, v in preds.items()}


def main():
    name, R, seed0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rows = []
    for r in range(R):
        try:
            rows.append({"seed": seed0 + r, "ok": True, **one(name, seed0 + r)})
        except Exception as e:  # recorded, never hidden
            rows.append({"seed": seed0 + r, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        print(name, rows[-1], flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"gamfit_replicates_{name}.json").write_text(json.dumps({"name": name, "rows": rows}))
    full = [row for row in rows if row["ok"] and "marginal" in row]
    rel = np.array([row["marginal"] / row["conditional"] - 1 for row in full])
    sd = rel.std(ddof=1) if len(rel) > 1 else 0.0
    t = rel.mean() / (sd / np.sqrt(len(rel))) if sd > 0 else 0.0
    print(f"SUMMARY {name} marginal vs conditional: mean rel change {100 * rel.mean():+.3f}%  "
          f"paired t={t:+.2f}  (R={len(rel)} of {len(rows)})")


if __name__ == "__main__":
    main()
