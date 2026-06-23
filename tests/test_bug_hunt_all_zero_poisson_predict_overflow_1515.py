"""Regression for #1515: all-zero Poisson predict() must return finite means.

An all-zero-count Poisson GAM produces a near-flat likelihood with an
astronomically wide posterior SE. The posterior-integrated mean
E[exp(η)] = exp(η + se²/2) previously overflowed to +inf (serialized
as JSON null / Python None), crashing predict() with a TypeError.

The fix floors the exponent when it would overflow, falling back to the
plug-in exp(η) — a finite, non-negative near-zero rate.
"""

import numpy as np
import gamfit


def test_all_zero_poisson_predict_returns_finite_mean():
    data = {"x": np.linspace(0.0, 1.0, 200), "y": np.zeros(200)}
    m = gamfit.fit(data, "y ~ s(x)", family="poisson")
    pred = np.asarray(m.predict(data), dtype=float).ravel()
    assert np.all(np.isfinite(pred)), f"predict must be finite, got non-finite values"
    assert np.all(pred >= 0.0), f"Poisson mean must be non-negative"
    assert pred.max() < 1.0, f"all-zero counts must give near-zero rate, got max={pred.max()}"


def test_all_zero_poisson_intercept_only_predict_returns_finite_mean():
    data = {"x": np.linspace(0.0, 1.0, 200), "y": np.zeros(200)}
    m = gamfit.fit(data, "y ~ 1", family="poisson")
    pred = np.asarray(m.predict(data), dtype=float).ravel()
    assert np.all(np.isfinite(pred)), f"predict must be finite"
    assert np.all(pred >= 0.0)
    assert pred.max() < 1.0, f"all-zero counts must give near-zero rate, got max={pred.max()}"


def test_all_zero_negative_binomial_predict_returns_finite_mean():
    data = {"x": np.linspace(0.0, 1.0, 200), "y": np.zeros(200)}
    m = gamfit.fit(data, "y ~ 1", family="negative_binomial")
    pred = np.asarray(m.predict(data), dtype=float).ravel()
    assert np.all(np.isfinite(pred)), f"predict must be finite"
    assert np.all(pred >= 0.0)
    assert pred.max() < 1.0, f"all-zero counts must give near-zero rate, got max={pred.max()}"
