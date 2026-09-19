"""pyGAM oracle: inference invariants (pygam/tests/test_GAM_methods.py).

Translated from pyGAM's interval, p-value and log-likelihood tests, keeping
only the statistically meaningful invariants:

* nested credible intervals: the 0.5 band lies inside the 0.9 band, which lies
  inside the 0.95 band, all around the reported posterior mean; the
  observation interval contains the credible one
  (test_conf_intervals_*, test_prediction_interval_*);
* a pure-noise covariate is not significant and a real signal is
  (test_pvalue_sig_impt);
* p-values and edf are invariant to rescaling a Gaussian response by 1e6
  (test_pvalue_invariant_to_scale);
* log-likelihood ordering saturated >= fit >= null, with the saturated and
  null values computed here in closed form (test_loglikelihood).

No calibration assertions: those belong to the p-value lane. pyGAM is never
imported.
"""

from __future__ import annotations

from math import lgamma
from typing import Any

import numpy as np
import pytest

import gamfit

LEVELS = (0.5, 0.9, 0.95)


def _gaussian_data(seed: int, n: int = 400) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    y = 1.0 + np.sin(6.0 * x) + rng.normal(0.0, 0.5, n)
    return {"x": x, "z": z, "y": y}


def _poisson_data(seed: int, n: int = 400) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    y = rng.poisson(np.exp(0.5 + np.sin(6.0 * x) + 0.3 * z)).astype(float)
    return {"x": x, "z": z, "y": y}


def _row(model: Any, covariate: str) -> dict[str, Any]:
    rows = [r for r in model.summary().smooth_terms if r["name"] == f"s({covariate})"]
    assert len(rows) == 1, model.summary().smooth_terms
    return dict(rows[0])


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_credible_intervals_nest_across_levels(family: str) -> None:
    data = _gaussian_data(40) if family == "gaussian" else _poisson_data(40)
    m = gamfit.fit(data, "y ~ s(x) + s(z)", family=family)
    grid = {"x": np.linspace(0.02, 0.98, 30), "z": np.linspace(0.98, 0.02, 30)}
    bands = {
        level: m.predict(grid, interval=level, observation_interval=True) for level in LEVELS
    }
    mean = np.asarray(bands[0.95]["posterior_mean"], float)
    for level in LEVELS:
        # The point prediction does not depend on the requested coverage.
        np.testing.assert_array_equal(np.asarray(bands[level]["posterior_mean"]), mean)
    lo = {lv: np.asarray(b["posterior_mean_lower"], float) for lv, b in bands.items()}
    hi = {lv: np.asarray(b["posterior_mean_upper"], float) for lv, b in bands.items()}
    # Strict nesting: a wider coverage must give a strictly wider band.
    assert np.all(lo[0.95] < lo[0.9]) and np.all(lo[0.9] < lo[0.5])
    assert np.all(lo[0.5] < mean) and np.all(mean < hi[0.5])
    assert np.all(hi[0.5] < hi[0.9]) and np.all(hi[0.9] < hi[0.95])
    for level, band in bands.items():
        # An interval for a new observation adds the response noise to the
        # posterior uncertainty of its mean, so it contains the credible band.
        assert np.all(np.asarray(band["observation_lower"]) < lo[level]), level
        assert np.all(np.asarray(band["observation_upper"]) > hi[level]), level


def test_noise_covariate_is_not_significant_and_signal_is() -> None:
    rng = np.random.default_rng(41)
    n = 1000
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    y = np.sin(6.0 * x) + rng.normal(0.0, 0.5, n)
    m = gamfit.fit({"x": x, "z": z, "y": y}, "y ~ s(x) + s(z)")
    assert float(_row(m, "z")["p_value"]) > 0.05
    assert float(_row(m, "x")["p_value"]) < 1e-6


@pytest.mark.parametrize("factor", [1e6, 1e-6])
def test_pvalues_and_edf_invariant_to_response_scale(factor: float) -> None:
    """Rescaling a Gaussian y rescales the fit and nothing else: REML profiles
    the dispersion, so lambda, edf and every test statistic are scale-free."""
    rng = np.random.default_rng(42)
    n = 500
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    y = np.sin(6.0 * x) + 0.4 * z + rng.normal(0.0, 0.5, n)
    base = {"x": x, "z": z, "y": y}
    scaled = {"x": x, "z": z, "y": factor * y}
    m1 = gamfit.fit(base, "y ~ s(x) + s(z)")
    m2 = gamfit.fit(scaled, "y ~ s(x) + s(z)")
    for covariate in ("x", "z"):
        r1, r2 = _row(m1, covariate), _row(m2, covariate)
        assert float(r2["edf"]) == pytest.approx(float(r1["edf"]), rel=1e-4), covariate
        assert float(r2["p_value"]) == pytest.approx(float(r1["p_value"]), rel=1e-3), covariate
    assert float(m2.summary().edf_total) == pytest.approx(float(m1.summary().edf_total), rel=1e-4)
    np.testing.assert_allclose(
        np.asarray(m2.predict(scaled)) / factor, np.asarray(m1.predict(base)), rtol=1e-6
    )


def _poisson_loglik(y: np.ndarray, mu: np.ndarray) -> float:
    ll = 0.0
    for yi, mi in zip(y, mu):
        ll += (yi * np.log(mi) if yi > 0 else 0.0) - mi - lgamma(yi + 1.0)
    return float(ll)


def test_poisson_loglik_ordering_saturated_fit_null() -> None:
    d = _poisson_data(43)
    y = d["y"]
    # mu_i = y_i; the y_i = 0 rows contribute exactly 0 (0 log 0 - 0 - log 0!).
    saturated = float(sum((yi * np.log(yi) if yi > 0 else 0.0) - yi - lgamma(yi + 1.0) for yi in y))
    null = _poisson_loglik(y, np.full_like(y, y.mean()))

    null_model = gamfit.fit(d, "y ~ 1", family="poisson")
    # The intercept-only MLE is the sample mean; its log-likelihood is exact.
    assert float(null_model.summary().log_likelihood) == pytest.approx(null, rel=1e-10)

    fit = float(gamfit.fit(d, "y ~ s(x) + s(z)", family="poisson").summary().log_likelihood)
    assert saturated > fit > null
    # The signal is strong: the fit recovers most of the saturated-vs-null gap.
    assert (fit - null) > 0.5 * (saturated - null)


def test_bernoulli_loglik_ordering_saturated_fit_null() -> None:
    rng = np.random.default_rng(44)
    n = 600
    x = rng.uniform(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-(2.0 * np.sin(6.0 * x))))
    y = (rng.uniform(size=n) < p).astype(float)
    d = {"x": x, "y": y}
    saturated = 0.0  # every mu_i = y_i in {0, 1}
    pbar = float(y.mean())
    null = float(n * (pbar * np.log(pbar) + (1.0 - pbar) * np.log(1.0 - pbar)))

    null_model = gamfit.fit(d, "y ~ 1", family="binomial")
    assert float(null_model.summary().log_likelihood) == pytest.approx(null, rel=1e-8)

    fit = float(gamfit.fit(d, "y ~ s(x)", family="binomial").summary().log_likelihood)
    assert saturated > fit > null


def test_gaussian_nested_fit_ordering() -> None:
    """pyGAM's loglik test on sine data: s(x) beats linear(x) beats the mean.

    For a Gaussian the residual sum of squares carries the ordering; the
    intercept-only and linear fits are closed-form OLS here."""
    rng = np.random.default_rng(45)
    n = 400
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(6.0 * x) + rng.normal(0.0, 0.3, n)
    d = {"x": x, "y": y}
    rss = {f: float(np.sum((y - gamfit.fit(d, f).predict(d)) ** 2)) for f in ("y ~ 1", "y ~ linear(x)", "y ~ s(x)")}
    np.testing.assert_allclose(rss["y ~ 1"], float(np.sum((y - y.mean()) ** 2)), rtol=1e-10)
    slope, icept = np.polyfit(x, y, 1)
    np.testing.assert_allclose(rss["y ~ linear(x)"], float(np.sum((y - (icept + slope * x)) ** 2)), rtol=1e-8)
    assert rss["y ~ s(x)"] < rss["y ~ linear(x)"] < rss["y ~ 1"]
    ll = {f: float(gamfit.fit(d, f).summary().log_likelihood) for f in rss}
    assert ll["y ~ s(x)"] > ll["y ~ linear(x)"] > ll["y ~ 1"]
