"""Diagnostic only (never shipped): paired replicate study of the full Laplace
rho-marginal predictive mean against the plug-in, on independent synthetic
datasets from the bench_accuracy 1-D generators.

Each replicate draws a fresh dataset, fits the full_laplace_folds REML/LAML model on
all of it, and scores truth-MSE on a 400-point grid over the covariate range
for plugin / first (first-order V_p) / gh (15^r Gauss-Hermite over N(rho_hat,
V_rho), including the E[beta_hat(rho)] - beta_hat mean shift). Replicates are
independent, so the paired t over R replicates is a valid test, unlike
overlapping CV folds.

Usage: python full_laplace_replicates.py GENERATOR R SEED0
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import full_laplace_folds as P  # noqa: E402


def gen(name, rng):
    if name == "sin1_n100":
        x = rng.uniform(0, 1, 100); f = lambda t: np.sin(2 * np.pi * t)
        sd = 0.3 * np.std(f(np.linspace(0, 1, 10001))) + 0.1
        return "gaussian", x, f(x) + rng.normal(0, sd, 100), f
    if name == "outlier_n300":
        n = 300; x = rng.uniform(0, 1, n); f = lambda t: np.sin(2 * np.pi * 2 * t)
        y = f(x) + rng.normal(0, 0.3, n); idx = rng.choice(n, n // 20, replace=False)
        y[idx] += rng.standard_t(1.5, len(idx)) * 5
        return "gaussian", x, y, f
    if name == "hetero_n500":
        n = 500; x = rng.uniform(0, 1, n); f = lambda t: np.sin(2 * np.pi * 2 * t) + t
        return "gaussian", x, f(x) + rng.normal(0, 0.05 + 1.2 * x ** 2, n), f
    if name == "binom_sin2_n500":
        n = 500; x = rng.uniform(0, 1, n); f = lambda t: 1 / (1 + np.exp(-2 * np.sin(2 * np.pi * 2 * t)))
        return "binomial", x, (rng.uniform(size=n) < f(x)).astype(float), f
    if name == "pois_lowcount_n500":
        n = 500; x = rng.uniform(0, 1, n); f = lambda t: np.exp(-1.5 + 1.5 * np.sin(2 * np.pi * 2 * t))
        return "poisson", x, rng.poisson(f(x)).astype(float), f
    if name == "coal_like_n150":
        # annual counts on a smooth, mostly flat-then-falling rate (coal-mining disasters shape)
        n = 150; x = np.sort(rng.uniform(0, 1, n)); f = lambda t: np.exp(1.0 - 2.2 / (1 + np.exp(-12 * (t - 0.35))))
        return "poisson", x, rng.poisson(f(x)).astype(float), f
    raise KeyError(name)


def one(fam, x, y, f):
    design, Sk = P.build(x); Xd = design(x)
    xg = np.linspace(x.min(), x.max(), 400); Xg = design(xg); mt = f(xg)
    m = P.Model(fam, Xd, y, Sk)
    obj = lambda r: m.fit(r)[0]
    rh = min((minimize(obj, np.array(s0, float), method="Nelder-Mead",
                       options=dict(xatol=1e-7, fatol=1e-10, maxiter=4000))
              for s0 in [(0, 0), (8, 8), (-3, 5), (12, 0), (4, -4)]), key=lambda o: o.fun).x
    rh = np.clip(rh, -15, 25)
    _, b0, C0 = m.fit(rh)
    evr, Ur = np.linalg.eigh(P.fd_hess(obj, rh))
    act = evr > 1e-8
    Vr = (Ur[:, act] / evr[act]) @ Ur[:, act].T
    J = np.column_stack([(m.fit(rh + 1e-4 * e)[1] - m.fit(rh - 1e-4 * e)[1]) / 2e-4 for e in np.eye(2)])
    m.fit(rh)
    q = lambda C: np.einsum("ij,jk,ik->i", Xg, C, Xg)
    out = {"plugin": P.mean_given(fam, Xg @ b0, q(C0)),
           "first": P.mean_given(fam, Xg @ b0, q(C0 + J @ Vr @ J.T))}
    gx, gw = np.polynomial.hermite_e.hermegauss(15); gw = gw / gw.sum()
    ks = np.where(act)[0]
    if len(ks) == 0:
        gx, gw = np.zeros(1), np.ones(1)
    nk = max(len(ks), 1)
    nodes = np.stack(np.meshgrid(*[gx] * nk, indexing="ij"), -1).reshape(-1, nk)
    wn = np.prod(np.stack(np.meshgrid(*[gw] * nk, indexing="ij"), -1).reshape(-1, nk), 1)
    acc = 0
    for z, wz in zip(nodes, wn):
        r = rh + sum(z[i] * Ur[:, k] / np.sqrt(evr[k]) for i, k in enumerate(ks))
        _, b, C = m.fit(r)
        acc = acc + wz * P.mean_given(fam, Xg @ b, q(C))
    m.fit(rh)
    out["gh"] = acc
    return {k: float(np.mean((v - mt) ** 2)) for k, v in out.items()}, int(act.sum())


if __name__ == "__main__":
    name, R, seed0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rows = []
    for r in range(R):
        fam, x, y, f = gen(name, np.random.default_rng(seed0 + r))
        res, nact = one(fam, x, y, f)
        rows.append({**res, "n_active": nact})
        print(name, r, {k: f"{v:.5g}" for k, v in res.items()}, nact, flush=True)
    json.dump({"name": name, "rows": rows},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"full_laplace_replicates_{name}.json"), "w"))
    for v in ("first", "gh"):
        rel = np.array([row[v] / row["plugin"] - 1 for row in rows])
        t = rel.mean() / (rel.std(ddof=1) / np.sqrt(len(rel))) if rel.std() > 0 else 0.0
        print(f"SUMMARY {name} {v}: mean rel change {100 * rel.mean():+.3f}%  paired t={t:+.2f}  (R={len(rel)})")
