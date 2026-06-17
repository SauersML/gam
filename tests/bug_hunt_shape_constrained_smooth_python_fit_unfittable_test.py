"""Regression test for #1191.

Every advertised shape-constrained smooth — `shape=monotone_increasing`,
`monotone_decreasing`, `convex`, `concave` — was *unfittable* through
`gamfit.fit` on ordinary textbook data. The fit aborted with

    IntegrationError: REML smoothing optimization failed to converge: no
    candidate seeds passed outer startup validation (standard REML)
      ... rejected_by_domain ... ALO exact frozen-curvature solve failed at
      row 0: non-finite score/curvature ...

even though the README advertises these exact calls and the `gam` CLI fits the
identical data fine.

Root cause: the ALO-stabilization augmentation in the REML outer objective is
an OUTER-OPTIMIZER aid (a leverage barrier), never part of the genuine
criterion, yet it propagated any failure of its exact frozen-curvature
leave-one-out diagnostic as a fatal error. Shape-constrained smooths pass
through an *indefinite* penalized Hessian at intermediate smoothing parameters,
which makes that diagnostic non-finite; the seed loop then classified the error
as a domain rejection and rejected every candidate seed.

With the fix the augmentation degrades gracefully (skips itself for that ρ and
falls back to the plain REML criterion), so the genuine criterion — which fits
this data fine — drives the search to a well-conditioned optimum. This test
fits the issue's exact deterministic data and asserts the constrained fit
returns a model whose prediction obeys the requested constraint on a held-out
grid, with an unconstrained `s(x)` control that already succeeds.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


def _sqrt_dataset(seed: int = 11, n: int = 400) -> pd.DataFrame:
    """Deterministic ``y = sqrt(x) + noise`` — strictly increasing AND concave."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = np.sqrt(x) + rng.normal(0.0, 0.05, n)
    return pd.DataFrame({"x": x, "y": y})


def _predict_on_grid(model: Any, df: pd.DataFrame, n_grid: int = 200) -> np.ndarray:
    grid = pd.DataFrame({"x": np.linspace(df["x"].min(), df["x"].max(), n_grid)})
    return np.asarray(model.predict(grid), dtype=float)


def test_unconstrained_control_fits() -> None:
    """Control: the unconstrained smooth fits and spans the response."""
    df = _sqrt_dataset()
    model = gamfit.fit(df, "y ~ s(x)")
    pred = _predict_on_grid(model, df)
    assert np.all(np.isfinite(pred))
    assert pred.max() - pred.min() > 0.5


def test_monotone_increasing_smooth_is_fittable_and_monotone() -> None:
    """The headline repro from the issue: this used to raise IntegrationError."""
    df = _sqrt_dataset()
    model = gamfit.fit(df, "y ~ s(x, shape=monotone_increasing)")
    pred = _predict_on_grid(model, df)
    assert np.all(np.isfinite(pred)), "prediction must be finite"
    # The defining property of the constraint: non-decreasing on a held-out grid.
    diffs = np.diff(pred)
    assert np.all(diffs >= -1e-6), f"prediction not monotone: min step {diffs.min():.3e}"
    # And a genuine positive rise — not collapsed to a constant.
    assert pred[-1] - pred[0] > 0.25, f"prediction did not rise: {pred[-1] - pred[0]:.4f}"


def test_monotone_decreasing_smooth_is_fittable_and_monotone() -> None:
    df = _sqrt_dataset()
    model = gamfit.fit(df, "y ~ s(x, shape=monotone_decreasing)")
    pred = _predict_on_grid(model, df)
    assert np.all(np.isfinite(pred)), "prediction must be finite"
    diffs = np.diff(pred)
    assert np.all(diffs <= 1e-6), f"prediction not non-increasing: max step {diffs.max():.3e}"


def test_convex_smooth_is_fittable_and_convex() -> None:
    df = _sqrt_dataset()
    model = gamfit.fit(df, "y ~ s(x, shape=convex)")
    pred = _predict_on_grid(model, df)
    assert np.all(np.isfinite(pred)), "prediction must be finite"
    second_diff = np.diff(pred, 2)
    span = pred.max() - pred.min()
    tol = 1e-5 * max(span, 1.0)
    assert np.all(second_diff >= -tol), f"prediction not convex: min curvature {second_diff.min():.3e}"


def test_concave_smooth_is_fittable_and_concave() -> None:
    df = _sqrt_dataset()
    model = gamfit.fit(df, "y ~ s(x, shape=concave)")
    pred = _predict_on_grid(model, df)
    assert np.all(np.isfinite(pred)), "prediction must be finite"
    second_diff = np.diff(pred, 2)
    span = pred.max() - pred.min()
    tol = 1e-5 * max(span, 1.0)
    assert np.all(second_diff <= tol), f"prediction not concave: max curvature {second_diff.max():.3e}"
    # Concave constraint is non-binding on sqrt data — fit must still span response.
    assert pred.max() - pred.min() > 0.25
