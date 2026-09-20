"""gam#3310 / gam#3366: the single-smooth closed-form Gaussian REML fit never
returns a zero fit with λ = NaN for a singular XᵀWX. An identified model (the
penalty is positive definite on null(W½X)) is fit exactly through the penalty
pencil (#3366); an unidentified one is refused with a typed error (#3310)."""

import numpy as np
import pytest

import gamfit.errors
from gamfit._api import gaussian_reml_fit


def _second_difference_penalty(p: int) -> np.ndarray:
    d = np.diff(np.eye(p), n=2, axis=0)
    return d.T @ d


def _dense_reml(x, y, penalty, lam):
    """The REML objective on the full, positive-definite H = XᵀX + λS."""
    n, p = x.shape
    h = x.T @ x + lam * penalty
    beta = np.linalg.solve(h, x.T @ y)
    s_eig = np.linalg.eigvalsh(penalty)
    positive = s_eig[s_eig > 1e-9 * np.abs(s_eig).max()]
    nu = n - (p - positive.size)
    score = 0.5 * y.shape[1] * (
        np.linalg.slogdet(h)[1] - np.log(positive).sum() - positive.size * np.log(lam)
    )
    for j in range(y.shape[1]):
        r = y[:, j] - x @ beta[:, j]
        dp = r @ r + lam * beta[:, j] @ penalty @ beta[:, j]
        score += 0.5 * nu * (1.0 + np.log(2.0 * np.pi * dp / nu))
    return score, beta


def _smooth_response(n: int, seed: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    rng = np.random.default_rng(seed)
    return (np.sin(2.3 * t) + 0.4 * np.sin(11.0 * t + 0.7) + 0.05 * rng.standard_normal(n))[
        :, None
    ]


def _assert_matches_dense(x, y, penalty):
    out = gaussian_reml_fit(x, y, penalty)
    lam = float(np.asarray(out["lambda"], dtype=float))
    assert np.isfinite(lam) and lam > 0.0
    score, beta = _dense_reml(x, y, penalty, lam)
    np.testing.assert_allclose(
        np.asarray(out["coefficients"], dtype=float), beta, rtol=0.0, atol=1e-8 * np.abs(beta).max()
    )
    assert abs(float(out["reml_score"]) - score) <= 1e-9 * (1.0 + abs(score))


def test_more_coefficients_than_rows_is_fit_through_the_pencil_3366():
    n, p = 12, 20
    t = np.linspace(0.0, 1.0, n)[:, None]
    knots = np.linspace(0.0, 1.0, p)[None, :]
    x = np.exp(-((t - knots) ** 2) / 0.02)
    _assert_matches_dense(x, _smooth_response(n, 3310), _second_difference_penalty(p))


def test_a_column_no_row_reaches_is_fit_through_the_pencil_3366():
    # A basis function no row reaches (the #3214 fixture's knots gave such
    # columns): XᵀWX has an exactly zero row and column, and the penalty alone
    # identifies that coefficient.
    n, p = 40, 6
    t = np.linspace(-1.0, 1.0, n)[:, None]
    x = t ** np.arange(p)[None, :]
    x[:, 3] = 0.0
    _assert_matches_dense(x, _smooth_response(n, 3311), _second_difference_penalty(p))


def test_a_penalty_blind_to_the_data_null_space_is_refused_3310():
    n, p = 40, 6
    t = np.linspace(-1.0, 1.0, n)[:, None]
    x = t ** np.arange(p)[None, :]
    x[:, 3] = 0.0
    penalty = _second_difference_penalty(p)
    penalty[3, :] = 0.0
    penalty[:, 3] = 0.0
    with pytest.raises(gamfit.errors.IllConditionedError):
        gaussian_reml_fit(x, _smooth_response(n, 3312), penalty)


def test_full_rank_design_still_fits_3310():
    rng = np.random.default_rng(3312)
    n, p = 40, 6
    x = rng.standard_normal((n, p))
    y = x @ rng.standard_normal((p, 1)) + 0.1 * rng.standard_normal((n, 1))
    out = gaussian_reml_fit(x, y, _second_difference_penalty(p))
    assert np.all(np.isfinite(np.asarray(out["lambda"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(out["coefficients"], dtype=float)))
