"""gam#3310: the single-smooth closed-form Gaussian REML fit whitens by XᵀWX,
so a design whose XᵀWX is singular (more coefficients than rows, or a
rank-deficient design) is refused with a typed error instead of returning a
zero fit with λ = NaN."""

import numpy as np
import pytest

import gamfit.errors
from gamfit._api import gaussian_reml_fit


def _second_difference_penalty(p: int) -> np.ndarray:
    d = np.diff(np.eye(p), n=2, axis=0)
    return d.T @ d


def test_more_coefficients_than_rows_is_refused_3310():
    rng = np.random.default_rng(3310)
    n, p = 12, 20
    x = rng.standard_normal((n, p))
    y = rng.standard_normal((n, 1))
    with pytest.raises(gamfit.errors.IllConditionedError):
        gaussian_reml_fit(x, y, _second_difference_penalty(p))


def test_rank_deficient_design_is_refused_3310():
    # A basis function no row reaches (the #3214 fixture's knots gave such
    # columns): XᵀWX has an exactly zero row and column.
    rng = np.random.default_rng(3311)
    n, p = 40, 6
    x = rng.standard_normal((n, p))
    x[:, 3] = 0.0
    y = rng.standard_normal((n, 1))
    with pytest.raises(gamfit.errors.IllConditionedError):
        gaussian_reml_fit(x, y, _second_difference_penalty(p))


def test_full_rank_design_still_fits_3310():
    rng = np.random.default_rng(3312)
    n, p = 40, 6
    x = rng.standard_normal((n, p))
    y = x @ rng.standard_normal((p, 1)) + 0.1 * rng.standard_normal((n, 1))
    out = gaussian_reml_fit(x, y, _second_difference_penalty(p))
    assert np.all(np.isfinite(np.asarray(out["lambda"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(out["coefficients"], dtype=float)))
