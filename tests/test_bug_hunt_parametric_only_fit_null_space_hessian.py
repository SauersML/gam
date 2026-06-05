"""Regression: a purely parametric (smooth-free) GLM must be fittable through
the documented Python API ``gamfit.fit``.

An ordinary linear model ``y ~ x1 + x2`` — no ``s()``/``te()``/``matern()``
smooth, just penalized linear terms — is the most basic model a GAM engine can
be asked for, and the CLI (``gam fit data.csv 'y ~ x1 + x2'``) fits it fine and
recovers the OLS coefficients. The README promises the CLI and ``gamfit`` "share
one engine"; the two should therefore agree on this trivial fit.

They do not. ``gamfit.fit(df, "y ~ x1 + x2")`` raises

    gamfit ... InvalidConfigurationError:
        null-space Hessian is not positive definite:
        Cholesky factorization failed: NonPositivePivot { index: 0 }

The *fit itself converges* — the abort happens afterwards, in the Python-only
payload-builder step ``build_standard_payload`` →
``fit_with_null_space_logdet`` → ``compute_null_space_metadata``
(``crates/gam-pyffi/src/lib.rs:29913-29991``). That step takes the penalty
null-space basis (``rrqr_nullspace_basis``, line 29969), restricts the fitted
penalized Hessian to it (line 29979-29981), and Cholesky-factorizes the result
to record a ``null_space_logdet`` for the saved payload (line 29982). For a
smooth-free design that restricted matrix is reported indefinite and the whole
``fit_table`` call fails. The CLI never runs this metadata path, which is why it
succeeds on identical data.

The bug is not family-specific: smooth-free Gaussian, binomial, Poisson and
Gamma models all fail the same way, while *adding a single smooth term* (e.g.
``y ~ s(x1) + x2``) makes the identical-otherwise model fit. The intercept-only
model ``y ~ 1`` also fits, so the trigger is "≥1 non-intercept parametric term
and no penalized smooth".

This test fits a deterministic linear-Gaussian dataset and asserts the fit
succeeds and tracks the closed-form OLS solution. It fails today at the
``gamfit.fit`` call; once the metadata step no longer rejects smooth-free
models it will pass unchanged.

Committed for the bug hunt; see the GitHub issue for the full write-up.
"""

from __future__ import annotations

import numpy as np

import gamfit


def _make_linear_dataset(n: int = 200):
    """Deterministic y = 1.5 + 2.0*x1 - 0.7*x2 + small noise."""
    rng = np.random.default_rng(20240605)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.3, n)
    y = 1.5 + 2.0 * x1 - 0.7 * x2 + noise
    return {"x1": x1, "x2": x2, "y": y}


def _ols_predictions(data) -> np.ndarray:
    n = len(data["y"])
    design = np.column_stack([np.ones(n), data["x1"], data["x2"]])
    beta, *_ = np.linalg.lstsq(design, data["y"], rcond=None)
    return design @ beta


def test_parametric_only_gaussian_fit_recovers_ols():
    data = _make_linear_dataset()

    # Fitting an ordinary linear model through the public Python API must not
    # raise. Today this raises InvalidConfigurationError ("null-space Hessian
    # is not positive definite") from the post-fit null-space-logdet metadata
    # step in gam-pyffi, even though the fit itself converged and the CLI fits
    # the same formula+data without error.
    model = gamfit.fit(data, "y ~ x1 + x2")

    preds = np.asarray(model.predict(data), dtype=float)
    ols = _ols_predictions(data)

    # A penalized linear fit with the default (tiny) ridge must reproduce the
    # OLS fitted values closely — the CLI fit of this exact dataset matches OLS
    # to < 2e-3 max abs deviation. Use a generous tolerance so the assertion is
    # about correctness, not the exact ridge strength.
    max_dev = float(np.max(np.abs(preds - ols)))
    assert max_dev < 1e-1, (
        f"parametric-only fit does not track the OLS solution: "
        f"max|pred - OLS| = {max_dev:.3e}"
    )

    # And it must explain the (essentially linear) response.
    truth = 1.5 + 2.0 * data["x1"] - 0.7 * data["x2"]
    ss_res = float(np.sum((preds - truth) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.99, f"parametric-only fit R^2 vs true linear signal = {r2:.4f}"
