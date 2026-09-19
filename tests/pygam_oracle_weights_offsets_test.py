"""pyGAM oracle: weights and offsets (pygam/tests/test_GAM_methods.py).

pyGAM checks that sample weights and exposure offsets are accepted and change
the fit. The exact statistical invariants are asserted here instead:

* integer Poisson weights are the same likelihood as duplicated rows, so the
  weighted fit equals the fit on the expanded data;
* multiplying every Gaussian weight by a constant is absorbed by the profiled
  dispersion, so the posterior mean and its standard error do not move;
* a log-exposure offset enters the Poisson predictor additively, so adding
  log(2) to the prediction offset doubles the predicted mean.

pyGAM is never imported.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def test_integer_poisson_weights_equal_row_duplication() -> None:
    rng = np.random.default_rng(80)
    n = 300
    x = rng.uniform(0.0, 1.0, n)
    y = rng.poisson(np.exp(1.0 + np.sin(6.0 * x))).astype(float)
    w = rng.integers(1, 4, n).astype(float)
    weighted = gamfit.fit({"x": x, "y": y, "w": w}, "y ~ s(x)", family="poisson", weights="w")
    reps = w.astype(int)
    expanded = gamfit.fit(
        {"x": np.repeat(x, reps), "y": np.repeat(y, reps)}, "y ~ s(x)", family="poisson"
    )
    grid = {"x": np.linspace(0.02, 0.98, 40)}
    np.testing.assert_allclose(
        np.asarray(weighted.predict(grid), float),
        np.asarray(expanded.predict(grid), float),
        rtol=1e-5,
    )
    assert float(weighted.summary().edf_total) == pytest.approx(
        float(expanded.summary().edf_total), rel=1e-4
    )


@pytest.mark.parametrize("factor", [1e-3, 7.0, 1e3])
def test_gaussian_weight_rescaling_is_absorbed_by_the_dispersion(factor: float) -> None:
    rng = np.random.default_rng(81)
    n = 300
    x = rng.uniform(0.0, 1.0, n)
    w = rng.uniform(0.5, 2.0, n)
    y = np.sin(6.0 * x) + rng.normal(0.0, 0.3, n) / np.sqrt(w)
    base = gamfit.fit({"x": x, "y": y, "w": w}, "y ~ s(x)", weights="w")
    scaled = gamfit.fit({"x": x, "y": y, "w": factor * w}, "y ~ s(x)", weights="w")
    grid = {"x": np.linspace(0.02, 0.98, 40)}
    p1 = base.predict(grid, interval=0.9)
    p2 = scaled.predict(grid, interval=0.9)
    np.testing.assert_allclose(
        np.asarray(p2["posterior_mean"], float), np.asarray(p1["posterior_mean"], float), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(p2["posterior_mean_standard_error"], float),
        np.asarray(p1["posterior_mean_standard_error"], float),
        rtol=1e-5,
    )


def _exposure_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(82)
    n = 500
    x = rng.uniform(0.0, 1.0, n)
    exposure = rng.uniform(0.5, 3.0, n)
    y = rng.poisson(exposure * np.exp(0.5 + np.sin(6.0 * x))).astype(float)
    return {"x": x, "y": y, "log_exposure": np.log(exposure)}


def test_log_exposure_offset_is_additive_on_the_predictor() -> None:
    d = _exposure_data()
    m = gamfit.fit(d, "y ~ s(x)", family="poisson", offset="log_exposure")
    ll = m.summary().log_likelihood
    assert ll is not None and np.isfinite(float(ll))

    grid = np.linspace(0.05, 0.95, 30)
    at_zero = {"x": grid, "log_exposure": np.zeros_like(grid)}
    doubled = {"x": grid, "log_exposure": np.full_like(grid, np.log(2.0))}
    r0 = m.predict(at_zero, interval=0.9)
    r2 = m.predict(doubled, interval=0.9)
    np.testing.assert_allclose(
        np.asarray(r2["linear_predictor_plugin"], float) - np.asarray(r0["linear_predictor_plugin"], float),
        np.log(2.0),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(r2["mean_plugin"], float), 2.0 * np.asarray(r0["mean_plugin"], float), rtol=1e-12
    )


def test_offset_is_not_absorbed_into_the_smooth() -> None:
    """Fitting with the exposure as an offset recovers the per-unit rate; a fit
    that ignored the offset would be biased by the mean log exposure."""
    d = _exposure_data()
    m = gamfit.fit(d, "y ~ s(x)", family="poisson", offset="log_exposure")
    grid = np.linspace(0.05, 0.95, 30)
    rate = np.asarray(m.predict({"x": grid, "log_exposure": np.zeros_like(grid)}), float)
    truth = np.exp(0.5 + np.sin(6.0 * grid))
    assert float(np.max(np.abs(np.log(rate) - np.log(truth)))) < 0.25
