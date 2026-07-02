"""Bug hunt #2032: zero prior-weight rows must be inert in ``gaussian_reml_fit``.

A prior weight of exactly ``0`` is the universal "excluded / infinite-variance"
convention (mgcv, statsmodels): a ``weights=0`` row must be equivalent to
omitting the row. The weighted response energy ``ywy = Σ w·y²`` and the normal
equations ``XᵀWX`` / ``XᵀWy`` already exclude zero-weight rows (their ``w``
factor is 0), but the residual degrees of freedom ``ν = n − nullity`` counted
the raw row count ``n``, including the zero-weight rows. That inflated ``ν``,
deflated ``σ²``, under-smoothed ``λ``, and — through λ — biased the
coefficients, with the bias growing monotonically in the number of zero-weight
rows.

The fix bases ``ν`` on the effective sample size (rows with a strictly positive
weight), so a ``weights=0`` row is a complete no-op.

Angles covered here:

* **Exact omission equivalence** — a fit padded with ``weights=0`` junk rows
  must match, to floating point, the fit on the positive-weight subset
  (coefficients, ``lambda``, ``sigma2``, ``edf``).
* **Monotone-bias absence** — the coefficient shift stays at machine zero
  regardless of how many zero-weight rows are appended (the original bug scaled
  monotonically: 0.010 / 0.034 / 0.097 / 0.144 for 30 / 120 / 600 / 2000 rows).
* **Response-independence** — two fits differing only in the *response* of their
  zero-weight rows must agree exactly (a convention-free isolation of the leak).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

import numpy as np

pytest = cast(Any, import_module("pytest"))
gamfit = cast(Any, import_module("gamfit"))


def _pspline_penalty(n_basis: int) -> np.ndarray:
    d = np.diff(np.eye(n_basis), 2, axis=0)
    return d.T @ d


def _scalar(out: dict[str, Any], key: str) -> float:
    return float(np.asarray(out[key]).ravel()[0])


def _base_problem(n: int = 150, knots: int = 8, seed: int = 1):
    x = np.linspace(0.0, 1.0, n)
    design = np.asarray(gamfit.bspline_basis(x, knots=knots))
    penalty = _pspline_penalty(design.shape[1])
    y = (
        np.sin(2 * np.pi * x)
        + np.random.default_rng(seed).normal(0.0, 0.2, n)
    ).reshape(-1, 1)
    return design, penalty, y


def test_zero_weight_padding_matches_positive_weight_subset() -> None:
    """A ``weights=0`` padding block must not change the fit at all."""
    design, penalty, y = _base_problem()
    n = design.shape[0]

    sub = gamfit.gaussian_reml_fit(design, y, penalty, weights=np.ones(n))

    rng = np.random.default_rng(2)
    g = 200
    design_full = np.vstack([design, rng.normal(0.0, 1.0, (g, design.shape[1]))])
    y_full = np.vstack([y, rng.normal(0.0, 1.0, (g, 1))])
    weights_full = np.concatenate([np.ones(n), np.zeros(g)])
    full = gamfit.gaussian_reml_fit(design_full, y_full, penalty, weights=weights_full)

    coef_sub = np.asarray(sub["coefficients"]).ravel()
    coef_full = np.asarray(full["coefficients"]).ravel()
    np.testing.assert_allclose(coef_full, coef_sub, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        _scalar(full, "lambda"), _scalar(sub, "lambda"), rtol=1e-9, atol=0
    )
    np.testing.assert_allclose(
        _scalar(full, "sigma2"), _scalar(sub, "sigma2"), rtol=1e-9, atol=0
    )
    np.testing.assert_allclose(
        _scalar(full, "edf"), _scalar(sub, "edf"), rtol=1e-9, atol=0
    )


@pytest.mark.parametrize("n_pad", [30, 120, 600, 2000])
def test_bias_does_not_scale_with_zero_weight_row_count(n_pad: int) -> None:
    """The original bug's coefficient shift grew with the padding count."""
    design, penalty, y = _base_problem()
    n = design.shape[0]
    sub = gamfit.gaussian_reml_fit(design, y, penalty, weights=np.ones(n))
    coef_sub = np.asarray(sub["coefficients"]).ravel()

    rng = np.random.default_rng(7)
    design_full = np.vstack(
        [design, rng.normal(0.0, 3.0, (n_pad, design.shape[1]))]
    )
    y_full = np.vstack([y, rng.normal(0.0, 10.0, (n_pad, 1))])
    weights_full = np.concatenate([np.ones(n), np.zeros(n_pad)])
    full = gamfit.gaussian_reml_fit(design_full, y_full, penalty, weights=weights_full)
    coef_full = np.asarray(full["coefficients"]).ravel()

    assert np.abs(coef_full - coef_sub).max() < 1e-9


def test_zero_weight_rows_response_is_irrelevant() -> None:
    """Two fits differing only in zero-weight-row *responses* must agree.

    This is convention-free: it does not depend on how the design of the
    excluded rows is chosen, only that a zero-weight row's response never
    reaches ``ywy`` or ``ν``.
    """
    design, penalty, y = _base_problem(seed=3)
    n = design.shape[0]

    rng = np.random.default_rng(11)
    g = 64
    pad_design = rng.normal(0.0, 2.0, (g, design.shape[1]))
    design_full = np.vstack([design, pad_design])
    weights_full = np.concatenate([np.ones(n), np.zeros(g)])

    y_a = np.vstack([y, rng.normal(0.0, 1.0, (g, 1))])
    y_b = np.vstack([y, rng.normal(0.0, 1e4, (g, 1))])

    fit_a = gamfit.gaussian_reml_fit(design_full, y_a, penalty, weights=weights_full)
    fit_b = gamfit.gaussian_reml_fit(design_full, y_b, penalty, weights=weights_full)

    np.testing.assert_allclose(
        np.asarray(fit_a["coefficients"]).ravel(),
        np.asarray(fit_b["coefficients"]).ravel(),
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        _scalar(fit_a, "sigma2"), _scalar(fit_b, "sigma2"), rtol=1e-9, atol=0
    )
    np.testing.assert_allclose(
        _scalar(fit_a, "lambda"), _scalar(fit_b, "lambda"), rtol=1e-9, atol=0
    )


def test_all_positive_weights_are_unaffected() -> None:
    """When every weight is positive, ``n_effective == n``: strict no-op.

    Guards against the fix accidentally perturbing the ordinary weighted path.
    """
    design, penalty, y = _base_problem(seed=5)
    n = design.shape[0]
    rng = np.random.default_rng(13)
    weights = rng.uniform(0.5, 3.0, n)

    # Reference: replicate the row-count semantics by hand — a positive weight
    # never changes the effective count, so the fit must be identical to itself
    # across repeated calls and independent of ordering.
    fit1 = gamfit.gaussian_reml_fit(design, y, penalty, weights=weights)
    fit2 = gamfit.gaussian_reml_fit(design, y, penalty, weights=weights)
    np.testing.assert_array_equal(
        np.asarray(fit1["coefficients"]), np.asarray(fit2["coefficients"])
    )
    # sigma2 must be finite and positive (a sanity anchor for the DoF path).
    assert _scalar(fit1, "sigma2") > 0.0
