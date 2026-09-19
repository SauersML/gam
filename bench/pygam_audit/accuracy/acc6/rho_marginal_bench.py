#!/usr/bin/env python3
"""ACC-6 evidence: does integrating the predictive mean over p(rho|y) improve
held-out accuracy anywhere in the bench_accuracy battery?

For every case and fold (same splits as bench_accuracy.py) this fits gamfit once
and scores three response-scale point predictions built from the fit's own
exported affine design and covariances (``Model.design_matrix``):

  plugin       g^{-1}(eta_hat)
  conditional  E[g^{-1}(eta)], eta ~ N(eta_hat, x' V_beta x)      (shipped default)
  marginal     E[g^{-1}(eta)], eta ~ N(eta_hat, x' V_p x)
               V_p = V_beta + J V_rho J' (gamfit's smoothing-corrected covariance)

``marginal`` is the first-order Laplace rho-marginal predictive mean
E_rho[E[mu|rho]]: to first order in the rho-perturbation the coefficient
posterior integrated over N(rho_hat, V_rho) is Gaussian with mean beta_hat
and covariance V_p (the mean shift E[beta_hat(rho)] - beta_hat is second
order), so the integrated mean is the inverse-link expectation under V_p.
For identity links all three coincide exactly.

Expectations use the closed form where it exists (log link: exp(eta + s^2/2)).
For the logit link they use the trapezoid rule on z in [-12, 12] with 8001
nodes: the integrand is analytic, so the rule converges geometrically, and the
truncated normal tail is below 1e-32. Against adaptive quadrature the error is
at most 5e-9 for s in [0.01, 20] and eta in [-15, 15]. Gauss-Hermite is not
used, because the logistic poles at eta + s z = +-i pi limit it to about 1e-3
at s = 8.

``shipped_vs_conditional_maxrel`` compares each fit's shipped
``posterior_mean`` with the ``conditional`` reconstruction, which shows the
reconstruction is exact.

Usage: python rho_marginal_bench.py [--only REGEX] [--skip REGEX]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # bench_accuracy.py lives one level up
import bench_accuracy as B  # noqa: E402
import gamfit  # noqa: E402

OUT = HERE / "results"

Z_NODES = np.linspace(-12.0, 12.0, 8001)
Z_WEIGHTS = np.exp(-0.5 * Z_NODES**2) / np.sqrt(2.0 * np.pi) * (Z_NODES[1] - Z_NODES[0])
Z_WEIGHTS[[0, -1]] *= 0.5


def expect_mean(family, eta, s2):
    if family in ("poisson", "gamma"):
        return np.exp(eta + 0.5 * s2)
    if family == "binomial":
        s = np.sqrt(np.maximum(s2, 0.0))
        z = eta[:, None] + s[:, None] * Z_NODES[None, :]
        return (0.5 * (1.0 + np.tanh(0.5 * z))) @ Z_WEIGHTS
    return eta.copy()


def inv_link(family, eta):
    if family in ("poisson", "gamma"):
        return np.exp(eta)
    if family == "binomial":
        return 1.0 / (1.0 + np.exp(-eta))
    return eta


def run_case(case, folds=5, seed=0):
    X = np.column_stack([case.cols[k] for k in case.cols]).astype(float)
    if case.family == "binomial":
        split = StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, case.y)
    else:
        split = KFold(folds, shuffle=True, random_state=seed).split(X)
    rows = []
    for k, (tr, te) in enumerate(split):
        names = list(case.cols)
        dtr = {c: case.cols[c][tr] for c in names}
        dtr["y"] = case.y[tr]
        dte = {c: case.cols[c][te] for c in names}
        mt = case.mu_true[te] if case.mu_true is not None else None
        rec = {"case": case.name, "fold": k}
        try:
            m = gamfit.fit(dtr, case.formula, family=case.family)
            shipped = np.asarray(m.predict(dte), float).ravel()
            a = m.design_matrix(dte)
            G = np.asarray(a.eta_gradient)
            eta = np.asarray(a.offset) + np.asarray(a.matrix) @ np.asarray(a.coefficients)
            Vc = a.covariance_conditional
            Vp = a.covariance_smoothing_corrected
            s2c = np.einsum("ij,jk,ik->i", G, Vc, G)
            preds = {"plugin": inv_link(case.family, eta),
                     "conditional": expect_mean(case.family, eta, s2c)}
            if Vp is not None:
                s2p = np.einsum("ij,jk,ik->i", G, Vp, G)
                preds["marginal"] = expect_mean(case.family, eta, s2p)
                rec["rel_var_increase_median"] = float(np.median((s2p - s2c) / np.maximum(s2c, 1e-300)))
            rec["shipped_vs_conditional_maxrel"] = float(
                np.max(np.abs(shipped - preds["conditional"]) / np.maximum(np.abs(shipped), 1e-300)))
            for name, mu in preds.items():
                rec[name] = B.metrics(case.family, case.y[te], mu, mt)
            rec["edf_total"] = float(m.summary().edf_total)
            rec["ok"] = True
        except Exception as e:  # recorded, never hidden
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rows.append(rec)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--skip")
    ap.add_argument("--no-big", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    cases = B.all_cases(big=not a.no_big)
    if a.only:
        cases = [c for c in cases if re.search(a.only, c.name)]
    if a.skip:
        cases = [c for c in cases if not re.search(a.skip, c.name)]
    for c in cases:
        out = OUT / f"{c.name}.json"
        if out.exists() and not a.force:
            continue
        rows = run_case(c)
        out.write_text(json.dumps({"name": c.name, "family": c.family, "n": int(len(c.y)),
                                   "synth": c.mu_true is not None, "rows": rows}))
        ok = [r for r in rows if r["ok"]]
        key = "truth_mse" if c.mu_true is not None else "dev"
        msg = " ".join(
            f"{v}={np.mean([r[v][key] for r in ok if v in r]):.6g}"
            for v in ("plugin", "conditional", "marginal") if ok and all(v in r for r in ok))
        print(f"{c.name:28s} {c.family:9s} ok={len(ok)}/{len(rows)} {key}: {msg}", flush=True)


if __name__ == "__main__":
    main()
