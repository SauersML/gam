"""Bug hunt: ``gam sample`` on a Beta-regression model ABORTS with
``NUTS sampling is not implemented for beta-regression logit`` instead of using
the documented Gaussian Laplace fallback that every other NUTS-unsupported
model class already uses.

The README and the CLI ``sample`` help both promise:

    "Posterior sampling uses NUTS over the coefficient posterior ... where the
     family supports it, and a Gaussian Laplace approximation otherwise."

The Laplace path exists and works — ``laplace_gaussian_fallback`` in
crates/gam-inference/src/sample.rs (line ~307) draws from ``N(mode, phi·H^{-1})``
using the saved penalised Hessian, and ``sample_saved_model`` (same file) routes
``GaussianLocationScale`` / ``BinomialLocationScale`` /
``DispersionLocationScale`` / ``BernoulliMarginalSlope`` / ``TransformationNormal``
to it. But a Beta GLM is a ``PredictModelClass::Standard`` model, and the
``Standard`` arm unconditionally calls ``sample_standard`` (the NUTS path):

    PredictModelClass::Standard => sample_standard(...)   // sample.rs:264-266

``sample_standard`` runs NUTS, which for a Beta-logit likelihood hits the hard
error at crates/gam-inference/src/hmc_io.rs:5442
(``"NUTS sampling is not implemented for beta-regression logit"``). There is no
``catch the unavailability -> laplace_gaussian_fallback`` branch in the
``Standard`` arm, so the whole command aborts (exit 1) rather than returning the
Laplace posterior.

Beta is the odd one out among the ``Standard`` families: Gaussian, Poisson,
Gamma, Tweedie, Negative-Binomial, and binomial logit/probit/cloglog all sample
successfully through this same path; only ``beta`` errors. Because every
downstream consumer (credible intervals, posterior-predictive checks,
``model.sample()``) depends on a working posterior surface, a Beta fit is left
with no posterior at all from the CLI.

This is a distinct subsystem from the Beta diagnose AIC/elpd precision bug
(posterior sampling vs the reporting log-likelihood); see ``Related`` in the
issue.

The test is fix-agnostic: it asserts only that ``gam sample`` on a Beta model
SUCCEEDS and returns a non-degenerate posterior (finite per-coefficient means,
strictly positive posterior std). Wiring the ``Standard``-arm NUTS-unavailable
path to ``laplace_gaussian_fallback`` (or implementing Beta NUTS) makes it pass
with no edit.

Driven through the ``gam`` CLI in the style of the other bug-hunt CLI tests.
"""

from __future__ import annotations

import csv
import math
import os
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


def test_beta_sample_uses_laplace_fallback_instead_of_aborting() -> None:
    # A plain, well-identified Beta GLM (no smooth needed to exhibit the bug).
    rng = np.random.default_rng(424242)
    n = 800
    x = rng.uniform(-2.0, 2.0, n)
    mu = 1.0 / (1.0 + np.exp(-(0.3 + 1.1 * x)))
    phi_true = 25.0
    y = np.clip(rng.beta(mu * phi_true, (1.0 - mu) * phi_true), 1e-4, 1.0 - 1e-4)

    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "beta.csv")
        model = os.path.join(d, "beta.gam")
        draws = os.path.join(d, "beta.posterior.csv")
        _write_csv(data, {"y": y, "x": x})

        fit = subprocess.run(
            [GAM_BIN, "fit", data, "y ~ x", "--family", "beta", "--out", model],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert fit.returncode == 0, f"beta fit failed:\n{fit.stdout}\n{fit.stderr}"

        samp = subprocess.run(
            [GAM_BIN, "sample", model, data, "--out", draws, "--samples", "200"],
            capture_output=True,
            text=True,
            timeout=600,
        )

        # Core assertion: posterior sampling must NOT abort for a Beta model.
        # Pre-fix this exits 1 with "NUTS sampling is not implemented for
        # beta-regression logit"; the documented contract is a Laplace fallback.
        assert samp.returncode == 0, (
            "gam sample aborted on a Beta model instead of using the documented "
            f"Gaussian Laplace fallback:\n{samp.stdout}\n{samp.stderr}"
        )
        assert os.path.exists(draws), "no posterior draws written"

        # The draws must be a real, finite posterior sample.
        with open(draws) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [r for r in reader if r]
        assert len(header) >= 1, "posterior draws have no coefficient columns"
        assert len(rows) >= 200, f"expected >=200 posterior draws, got {len(rows)}"
        cols = list(zip(*[[float(v) for v in r] for r in rows]))
        max_var = 0.0
        for j, col in enumerate(cols):
            assert all(math.isfinite(v) for v in col), f"non-finite draw in column {j}"
            mean = sum(col) / len(col)
            assert math.isfinite(mean), f"non-finite posterior mean in column {j}"
            var = sum((v - mean) ** 2 for v in col) / len(col)
            max_var = max(max_var, var)
        # A real posterior surface has genuine spread; a point mass would mean the
        # fallback never actually sampled.
        assert max_var > 0.0, (
            "degenerate posterior: every coefficient draw column has zero variance "
            "(a valid Laplace surface has strictly positive coefficient spread)"
        )
