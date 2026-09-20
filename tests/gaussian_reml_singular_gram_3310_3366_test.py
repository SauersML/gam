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


def _dense_reml(x, y, penalty, w, lam):
    """The REML objective on the full, positive-definite H = XᵀWX + λS."""
    n, p = x.shape
    h = x.T @ (w[:, None] * x) + lam * penalty
    beta = np.linalg.solve(h, x.T @ (w[:, None] * y))
    s_eig = np.linalg.eigvalsh(penalty)
    positive = s_eig[s_eig > 1e-9 * np.abs(s_eig).max()]
    nu = n - (p - positive.size)
    score = 0.5 * y.shape[1] * (
        np.linalg.slogdet(h)[1] - np.log(positive).sum() - positive.size * np.log(lam)
    )
    for j in range(y.shape[1]):
        r = y[:, j] - x @ beta[:, j]
        dp = r @ (w * r) + lam * beta[:, j] @ penalty @ beta[:, j]
        score += 0.5 * nu * (1.0 + np.log(2.0 * np.pi * dp / nu)) - 0.5 * np.log(w).sum()
    return score, beta


def _assert_matches_dense(x, y, penalty, w):
    out = gaussian_reml_fit(x, y, penalty, weights=w)
    lam = float(np.asarray(out["lambda"], dtype=float))
    assert np.isfinite(lam) and lam > 0.0
    score, beta = _dense_reml(x, y, penalty, w, lam)
    np.testing.assert_allclose(
        np.asarray(out["coefficients"], dtype=float), beta, rtol=0.0, atol=1e-8 * np.abs(beta).max()
    )
    assert abs(float(out["reml_score"]) - score) <= 1e-9 * (1.0 + abs(score))


def _zero_column_design():
    # A basis function no row reaches (the #3214 fixture's knots gave such
    # columns): XᵀWX has an exactly zero row and column.
    n, p = 24, 7
    t = (np.arange(n) - 11.5) / 12.0
    x = t[:, None] ** np.arange(p)[None, :] + 0.03 * np.cos(
        3.0 * np.arange(n)[:, None] + np.arange(p)[None, :]
    )
    x[:, 3] = 0.0
    y = (0.3 + 0.8 * t - 0.5 * t * t + 0.15 * np.sin(9.0 * t))[:, None]
    w = 1.0 + 0.1 * np.sin(0.5 * np.arange(n))
    return x, y, w


def test_more_coefficients_than_rows_is_fit_through_the_pencil_3366():
    n, p = 9, 12
    t = (np.arange(n) / (n - 1))[:, None]
    knots = (np.arange(p) / (p - 1))[None, :]
    x = np.exp(-((t - knots) ** 2) / 0.045) + 0.05 * np.sin(
        np.arange(n)[:, None] + 2.0 * np.arange(p)[None, :]
    )
    t = t[:, 0]
    y = (np.sin(2.3 * t) + 0.4 * np.sin(11.0 * t + 0.7))[:, None]
    w = 1.0 + 0.2 * np.cos(0.9 * np.arange(n))
    _assert_matches_dense(x, y, _second_difference_penalty(p), w)


def test_a_column_no_row_reaches_is_fit_through_the_pencil_3366():
    # The penalty alone identifies the coefficient of the unreached column.
    x, y, w = _zero_column_design()
    _assert_matches_dense(x, y, _second_difference_penalty(x.shape[1]), w)


def test_a_penalty_blind_to_the_data_null_space_is_refused_3310():
    x, y, w = _zero_column_design()
    penalty = _second_difference_penalty(x.shape[1])
    penalty[3, :] = 0.0
    penalty[:, 3] = 0.0
    with pytest.raises(gamfit.errors.IllConditionedError):
        gaussian_reml_fit(x, y, penalty, weights=w)


def test_full_rank_design_still_fits_3310():
    rng = np.random.default_rng(3312)
    n, p = 40, 6
    x = rng.standard_normal((n, p))
    y = x @ rng.standard_normal((p, 1)) + 0.1 * rng.standard_normal((n, 1))
    out = gaussian_reml_fit(x, y, _second_difference_penalty(p))
    assert np.all(np.isfinite(np.asarray(out["lambda"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(out["coefficients"], dtype=float)))
