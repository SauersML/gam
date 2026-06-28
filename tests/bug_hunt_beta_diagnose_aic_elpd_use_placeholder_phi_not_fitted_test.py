"""Bug hunt: ``gam diagnose`` reports the Beta-family AIC and PSIS-LOO ``elpd``
using the *placeholder* precision ``phi = 1.0`` instead of the fitted precision
``phi_hat`` the model actually estimated, making both channels wrong by a huge,
data-dependent constant (~1700 nats here) and incomparable to every other
family.

Root cause (a data-flow bug, NOT a missing normalizer):

* A Beta fit estimates the precision ``phi`` and stores it in the *scale*
  metadata ``fit_result.likelihood_scale.EstimatedBetaPhi.phi`` (≈ 22 for the
  data below). But it leaves the *family spec*
  ``fit_result.likelihood_family.response.Beta.phi`` at the placeholder ``1.0``.
* ``model_comparison_from_unified``
  (crates/gam-inference/src/model_comparison.rs) recomputes the fully-normalized
  reporting log-likelihood for the AIC / elpd by calling the per-row kernel in
  crates/gam-solve/src/pirls/deviance.rs. The Beta arm of that kernel
  (``calculate_loglikelihood`` line ~352 and ``pointwise_loglikelihood`` line
  ~459) destructures ``ResponseFamily::Beta { phi }`` and uses **that** ``phi``
  (the placeholder ``1.0``), ignoring ``likelihood.scale``'s
  ``EstimatedBetaPhi``. Contrast the Gaussian arm, which reads
  ``likelihood.scale.fixed_phi()``, and the Negative-Binomial fit, which DOES
  propagate its estimated ``theta`` into ``likelihood_family`` (so NB AIC/elpd
  are correct). Beta is the odd one out.

Consequence: for a concentrated Beta fit (large ``phi_hat``) the per-point Beta
log-density is large and positive, so the true in-sample log-likelihood is large
and positive (≈ +1521 below). Evaluated at the bogus ``phi = 1`` it is ≈ -202.
``gam diagnose`` reports the ``phi = 1`` value, yielding a nonsensical AIC and an
``elpd`` larger than the (also wrong) in-sample log-likelihood, i.e. a *negative*
effective number of parameters.

This is the Beta sibling of the elpd/AIC reporting bugs #1581 / #1582 / #1583
(which fixed dropped normalizers / the profiled-Gaussian scale) — none of those
threaded the *Beta* estimated precision into the reported kernel.

The test is fix-agnostic: it asserts only that the reported AIC and elpd are
absolute log-likelihoods evaluated at the model's OWN fitted precision
``phi_hat``, i.e. they must be far closer to the Beta log-likelihood at
``phi_hat`` than to the one at the placeholder ``phi = 1``. Either fix (thread
the estimated phi into the spec, or read the kernel's phi from the scale) makes
it pass with no edit.

Driven through the ``gam`` CLI (the buildable front-end whose
``model_comparison_from_unified`` assembles exactly these reported channels), in
the style of ``bug_hunt_diagnose_elpd_aic_reporting_normalizers_test.py``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import tempfile

import numpy as np
import pytest

GAM_BIN = os.path.join(os.path.dirname(__file__), "..", "target", "release", "gam")

pytestmark = pytest.mark.skipif(
    not os.path.exists(GAM_BIN),
    reason="release `gam` CLI binary not built (target/release/gam absent)",
)


def _write_csv(path: str, cols: dict[str, np.ndarray]) -> None:
    keys = list(cols)
    n = len(cols[keys[0]])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([cols[k][i] for k in keys])


def _run(*args: str) -> str:
    r = subprocess.run([GAM_BIN, *args], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"gam {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def _diagnose_metrics(model_json: str, data_csv: str) -> dict[str, float]:
    out = _run("diagnose", model_json, data_csv)
    metrics: dict[str, float] = {}
    for line in out.splitlines():
        if "AIC (conditional)" in line:
            m = re.search(r"(-?\d+\.\d+)", line)
            if m:
                metrics["aic"] = float(m.group(1))
        elif "edf (conditional)" in line:
            m = re.search(r"(-?\d+\.\d+)", line)
            if m:
                metrics["edf"] = float(m.group(1))
        elif "PSIS-LOO elpd" in line:
            m = re.search(r"(-?\d+\.\d+)", line)
            if m:
                metrics["elpd"] = float(m.group(1))
    return metrics


def _find_estimated_beta_phi(obj) -> float | None:
    """Recursively pull `likelihood_scale.EstimatedBetaPhi.phi` from the model."""
    if isinstance(obj, dict):
        if "EstimatedBetaPhi" in obj and isinstance(obj["EstimatedBetaPhi"], dict):
            return float(obj["EstimatedBetaPhi"]["phi"])
        for v in obj.values():
            r = _find_estimated_beta_phi(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_estimated_beta_phi(v)
            if r is not None:
                return r
    return None


def _beta_loglik(y: np.ndarray, mu: np.ndarray, phi: float) -> float:
    """Total fully-normalized Beta log-likelihood with mean `mu`, precision `phi`
    (a = mu*phi, b = (1-mu)*phi). Uses math.lgamma (stdlib, no scipy)."""
    total = 0.0
    lg = math.lgamma
    lphi = lg(phi)
    for yi, mui in zip(y.tolist(), mu.tolist()):
        a = mui * phi
        b = (1.0 - mui) * phi
        total += (
            lphi
            - lg(a)
            - lg(b)
            + (a - 1.0) * math.log(yi)
            + (b - 1.0) * math.log(1.0 - yi)
        )
    return total


def test_beta_diagnose_aic_elpd_use_fitted_phi_not_placeholder() -> None:
    # Concentrated Beta data: a smooth mean in (0,1) with a large precision, so
    # the fitted phi_hat is far from the placeholder 1.0 and the per-point Beta
    # log-density is large and positive — making the placeholder-phi error a
    # huge, unambiguous offset.
    rng = np.random.default_rng(20260628)
    n = 1200
    x = rng.uniform(0.0, 1.0, n)
    mu = 1.0 / (1.0 + np.exp(-(0.4 + 1.8 * np.sin(2.5 * x))))
    phi_true = 22.0
    a = mu * phi_true
    b = (1.0 - mu) * phi_true
    y = np.clip(rng.beta(a, b), 1e-4, 1.0 - 1e-4)

    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "beta.csv")
        model = os.path.join(d, "beta.gam")
        pred = os.path.join(d, "pred.csv")
        _write_csv(data, {"y": y, "x": x})

        _run("fit", data, "y ~ s(x)", "--family", "beta", "--out", model)

        metrics = _diagnose_metrics(model, data)
        assert "aic" in metrics and "elpd" in metrics and "edf" in metrics, (
            f"could not parse AIC/elpd/edf from diagnose: {metrics}"
        )
        aic = metrics["aic"]
        elpd = metrics["elpd"]
        edf = metrics["edf"]

        # The precision the model actually estimated (lives only in the scale
        # metadata; the family spec keeps the placeholder phi = 1.0).
        phi_hat = _find_estimated_beta_phi(json.load(open(model)))
        assert phi_hat is not None, "model is missing EstimatedBetaPhi"
        # The bug is only meaningful when phi_hat differs substantially from 1.
        assert phi_hat > 5.0, f"phi_hat={phi_hat} too close to the placeholder"

        # Fitted means (response scale) at the training rows.
        _run("predict", model, data, "--out", pred)
        mu_hat = np.array([float(r["mean"]) for r in csv.DictReader(open(pred))])

        # Two reference total log-likelihoods, evaluated at the SAME fitted means:
        #   L_hat  — at the fitted precision phi_hat (the correct reporting value)
        #   L_one  — at the placeholder phi = 1.0 (what the buggy kernel uses)
        L_hat = _beta_loglik(y, mu_hat, phi_hat)
        L_one = _beta_loglik(y, mu_hat, 1.0)

        # Sanity: the two references are far apart (the size of the dispersion
        # error), so "closer to L_hat than L_one" is a crisp, well-posed test.
        gap = abs(L_hat - L_one)
        assert gap > 500.0, f"references too close to discriminate (gap={gap:.1f})"

        # AIC = -2*logLik + 2*(edf + scale_dof); the +1 scale dof (estimated phi)
        # cancels in the distance comparison below, so the result is robust to
        # that convention.
        scale_dof = 1.0
        aic_at_phi_hat = -2.0 * L_hat + 2.0 * (edf + scale_dof)
        aic_at_phi_one = -2.0 * L_one + 2.0 * (edf + scale_dof)

        # --- Channel 1: conditional AIC must use the fitted precision ----------
        assert abs(aic - aic_at_phi_hat) < abs(aic - aic_at_phi_one), (
            "Beta conditional AIC is computed at the placeholder precision phi=1, "
            f"not the fitted phi_hat={phi_hat:.3f}: reported AIC={aic:.2f}, "
            f"AIC@phi_hat={aic_at_phi_hat:.2f}, AIC@phi=1={aic_at_phi_one:.2f}"
        )

        # --- Channel 2: PSIS-LOO elpd is an absolute log predictive density ----
        # It must also be evaluated at the fitted precision (it is reported in the
        # same units as the in-sample log-likelihood). The leave-one-out penalty
        # keeps elpd slightly BELOW L_hat, never ~1700 nats below at the phi=1
        # value.
        assert abs(elpd - L_hat) < abs(elpd - L_one), (
            "Beta PSIS-LOO elpd is computed at the placeholder precision phi=1, "
            f"not the fitted phi_hat={phi_hat:.3f}: reported elpd={elpd:.2f}, "
            f"L@phi_hat={L_hat:.2f}, L@phi=1={L_one:.2f}"
        )
