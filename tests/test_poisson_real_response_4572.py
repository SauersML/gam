"""gam#4572: an explicit Poisson family fits a non-negative real response.

Under a log link the Poisson log-likelihood in the coefficients is
``sum_i w_i (y_i eta_i - exp(eta_i))`` plus a term that depends on neither the
coefficients nor the smoothing parameters, so the fit depends on ``y`` only
through ``X' W y``. Any non-negative ``y*`` with the same ``X' W y*`` must
therefore give the same coefficients. That is the whole point of accepting a
non-integer response: pseudo-counts from iterative proportional fitting share a
table's sufficient statistics without being integers.

The same identity through prior weights: a row with response 1, prior weight
``w`` and offset ``log E - log w`` contributes ``w (x'b) - E exp(x'b)`` plus a
constant, exactly what response ``w`` with offset ``log E`` contributes. So a
fit carried entirely by fractional prior weights must reproduce the fit on the
response.

Agreement is judged in units of each coefficient's own standard error. Both
fits target one optimum, so their difference is the solvers' convergence
residual, which must be a vanishing fraction of the estimate's sampling
uncertainty. A millionth of a standard error is far below anything a user could
read off the fit, and far above double-precision rounding of the estimate.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit

N = 240
FORMULA = "y ~ x"


def _frame(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, N)
    exposure = rng.uniform(1.0e3, 5.0e3, N)
    y = rng.poisson(exposure * np.exp(-6.0 + np.sin(3.0 * x))).astype(float)
    return x, exposure, y


def _estimates(model: Any) -> tuple[np.ndarray, np.ndarray]:
    records = model.summary().coefficients
    estimate = np.asarray([record["estimate"] for record in records], dtype=float)
    std_error = np.asarray([record["std_error"] for record in records], dtype=float)
    assert np.all(np.isfinite(estimate)) and np.all(std_error > 0.0), records
    return estimate, std_error


def _same_sufficient_statistic(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """A non-integer ``y*`` with ``X' y* = X' y`` for the design ``[1, x]``,
    kept non-negative: a random direction projected off the design's column
    space, scaled so the shifted response stays at or above zero."""
    design = np.column_stack([np.ones_like(x), x])
    direction = np.random.default_rng(seed).standard_normal(len(x))
    coefficients, *_ = np.linalg.lstsq(design, direction, rcond=None)
    direction = direction - design @ coefficients
    negative = direction < 0.0
    room = np.min(y[negative] / -direction[negative]) if np.any(negative) else 1.0
    shifted = y + 0.5 * room * direction
    assert np.all(shifted >= 0.0)
    assert np.any(shifted != np.round(shifted)), "the fixture must carry non-integer counts"
    np.testing.assert_allclose(design.T @ shifted, design.T @ y, rtol=1e-12)
    return shifted


def test_a_real_response_sharing_x_prime_y_reproduces_the_integer_fit_4572() -> None:
    x, exposure, y = _frame(4572)
    base = gamfit.fit(
        {"x": x, "y": y, "off": np.log(exposure)}, FORMULA, family="poisson", offset="off"
    )
    y_star = _same_sufficient_statistic(x, y, seed=45721)
    shifted = gamfit.fit(
        {"x": x, "y": y_star, "off": np.log(exposure)}, FORMULA, family="poisson", offset="off"
    )
    beta, se = _estimates(base)
    beta_star, _ = _estimates(shifted)
    gap = np.abs(beta_star - beta) / se
    print(f"[4572] beta={beta} beta*={beta_star} gap/se={gap}")
    assert np.all(gap <= 1.0e-6), f"coefficients moved by {gap} standard errors"


def test_fractional_prior_weights_reproduce_the_response_fit_4572() -> None:
    x, exposure, y = _frame(4573)
    w = y + 0.5
    on_response = gamfit.fit(
        {"x": x, "y": w, "off": np.log(exposure)}, FORMULA, family="poisson", offset="off"
    )
    on_weights = gamfit.fit(
        {"x": x, "y": np.ones(N), "w": w, "off": np.log(exposure) - np.log(w)},
        FORMULA,
        family="poisson",
        offset="off",
        weights="w",
    )
    beta, se = _estimates(on_response)
    beta_w, _ = _estimates(on_weights)
    gap = np.abs(beta_w - beta) / se
    print(f"[4572] response beta={beta} weights beta={beta_w} gap/se={gap}")
    assert np.all(gap <= 1.0e-6), f"coefficients moved by {gap} standard errors"
