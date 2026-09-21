"""Public regressions for the binomial location-scale no-wiggle fit (#2358, #3879).

#2358: the no-wiggle pilot of a binomial location-scale fit used to be refused
by the outer LAML certificate even though the inner solve reached a valid mode.
The supported call must return a usable model.

#3879: binomial location-scale data see sigma only through the composite
``q = -threshold / sigma``, so a constant log-sigma level is the scale of the
threshold and not a parameter; the log-sigma design carries no intercept
(sigma = 1 where every noise covariate is zero). An intercept-only
``noise_formula="1"`` therefore has nothing left to fit and is refused, where it
used to be pinned by a coefficient-metric ridge.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit

from scipy.stats import norm


def _heteroskedastic_probit_data(rows: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2358)
    x = rng.uniform(-2.0, 2.0, rows)
    threshold = 0.3 - 0.8 * x
    probability = norm.cdf(-threshold * np.exp(-0.4 * x))
    return {"x": x, "y": (rng.random(rows) < probability).astype(float)}


def test_binomial_location_scale_no_wiggle_fit_mints() -> None:
    data = _heteroskedastic_probit_data(600)
    model = gamfit.fit(
        data,
        "y ~ x",
        family="binomial",
        link="probit",
        noise_formula="x",
    )
    prediction = np.asarray(model.predict(data), dtype=float)

    assert prediction.shape == (600,)
    assert np.all(np.isfinite(prediction))
    assert np.all((prediction >= 0.0) & (prediction <= 1.0))


def test_binomial_location_scale_intercept_only_noise_is_refused() -> None:
    rows = 60
    data = {
        "x": np.linspace(-2.0, 2.0, rows),
        "y": (np.arange(rows) % 2 == 0).astype(float),
    }
    with pytest.raises(
        gamfit.errors.GamfitError,
        match="intercept-only log_sigma is not a binomial scale model",
    ):
        gamfit.fit(data, "y ~ x", family="binomial", noise_formula="1")
