"""Bug hunt (#1047): a spline-scan-routed Gaussian smooth silently drops
``predict(observation_interval=True)``.

Since #1030 the single-1-D-smooth, Gaussian-identity shape auto-routes through
the exact O(n) state-space spline scan (#1044 added the order-3 quintic). For
any such scan-routed model ``predict(observation_interval=True)`` was silently
ignored: the payload carried the mean credible band but **no**
``observation_lower`` / ``observation_upper`` columns — while the dense path
emits both for the identical request.

The scan IS the exact Gaussian smoothing-spline posterior, so the predictive
(observation) interval is well defined::

    Var(y*) = Var(f(x*)) + sigma2

where ``Var(f(x*))`` is the off-knot posterior variance the bridge returns from
``SplineScanFit::predict`` and ``sigma2`` is the profiled Gaussian observation
variance (``SplineScanFit::sigma2``). The confidence band uses ``Var(f(x*))``
alone; the observation band inflates it by ``sigma2``. The fix threads
``observation_interval`` through the scan predict arm in
``crates/gam-pyffi/src/lib.rs`` (``predict_columns``).

This test fits both the cubic (``degree=3, penalty_order=2``) and quintic
(``degree=5, penalty_order=3``) scan forms, requests the observation interval,
and asserts the columns are present and non-empty, that the predictive band
strictly contains the mean credible band, and that its half-width reflects the
real ``sigma2`` (so a degenerate ``sigma2 == 0`` would fail) — roughly
``z * sigma`` wider per side, with ``sigma ~ 0.25`` for the DGP below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gamfit

# Quantile for the 95% two-sided normal interval used throughout.
_Z95 = 1.959963984540054


def _make_data(seed: int = 1, n: int = 500, noise: float = 0.25) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, noise, n)
    return pd.DataFrame({"x": x, "y": y})


# The two scan-routed forms #1044/#1030 divert to the O(n) state-space smoother:
# the cubic (penalty order 2) and quintic (penalty order 3). Both must honour
# the observation-interval request.
_SCAN_FORMULAS = [
    'y ~ s(x, bs="ps", degree=3, penalty_order=2, double_penalty=false)',
    'y ~ s(x, bs="ps", degree=5, penalty_order=3, double_penalty=false)',
]


def _grid() -> pd.DataFrame:
    return pd.DataFrame({"x": np.linspace(0.05, 0.95, 20)})


@pytest.mark.parametrize("formula", _SCAN_FORMULAS)
def test_scan_predict_emits_observation_interval_columns(formula: str) -> None:
    model = gamfit.fit(_make_data(), formula)
    out = model.predict(
        _grid(),
        interval=0.95,
        covariance_mode="conditional",
        observation_interval=True,
        return_type="dict",
    )
    cols = set(out)
    assert "observation_lower" in cols, (
        "spline-scan predict(observation_interval=True) dropped 'observation_lower' "
        f"silently; got columns {sorted(cols)}"
    )
    assert "observation_upper" in cols, (
        "spline-scan predict(observation_interval=True) dropped 'observation_upper' "
        f"silently; got columns {sorted(cols)}"
    )
    olo = np.asarray(out["observation_lower"], dtype=float)
    ohi = np.asarray(out["observation_upper"], dtype=float)
    assert olo.size == ohi.size > 0, "observation interval columns are empty"
    assert np.all(np.isfinite(olo)) and np.all(np.isfinite(ohi))


@pytest.mark.parametrize("formula", _SCAN_FORMULAS)
def test_scan_observation_band_strictly_wider_by_sigma2(formula: str) -> None:
    noise = 0.25
    model = gamfit.fit(_make_data(noise=noise), formula)
    out = model.predict(
        _grid(),
        interval=0.95,
        covariance_mode="conditional",
        observation_interval=True,
        return_type="dict",
    )
    mean = np.asarray(out["mean"], dtype=float)
    mlo = np.asarray(out["mean_lower"], dtype=float)
    mhi = np.asarray(out["mean_upper"], dtype=float)
    olo = np.asarray(out["observation_lower"], dtype=float)
    ohi = np.asarray(out["observation_upper"], dtype=float)

    tol = 1e-9
    # A prediction interval must bracket its own point prediction.
    assert np.all(olo <= mean + tol) and np.all(ohi >= mean - tol), (
        "observation band does not contain the point prediction"
    )
    # Var(y*) = Var(f) + sigma2 >= Var(f): the observation band is never narrower
    # than the mean credible band, and STRICTLY wider because sigma2 > 0.
    obs_width = ohi - olo
    mean_width = mhi - mlo
    assert np.all(obs_width >= mean_width - tol), (
        "observation interval narrower than the mean credible interval: "
        f"obs_width={obs_width}, mean_width={mean_width}"
    )
    # The inflation is real, not a degenerate sigma2 == 0 (which would make the
    # two bands identical and silently pass a >= check). Recover the implied
    # mean-side standard error from the credible band, add it in quadrature with
    # the profiled sigma, and require the observation half-width to match — this
    # pins sigma to the DGP noise (~0.25) and rejects sigma2 == 0.
    obs_half = 0.5 * obs_width
    mean_half = 0.5 * mean_width
    var_f = (mean_half / _Z95) ** 2
    # The observation half-width strictly exceeds the mean half-width everywhere.
    assert np.all(obs_half > mean_half + 1e-6), (
        "observation band is not strictly wider than the credible band; "
        "sigma2 appears to have been dropped (degenerate sigma2 == 0)"
    )
    # Implied observation variance minus the mean variance recovers a single,
    # positive, x-independent sigma2 (the profiled scale is one scalar).
    implied_sigma2 = (obs_half / _Z95) ** 2 - var_f
    assert np.all(implied_sigma2 > 0.0), (
        f"implied sigma2 not positive: {implied_sigma2}"
    )
    implied_sigma = np.sqrt(implied_sigma2)
    # sigma2 is a single profiled scalar: the recovered per-point sigma must be
    # (near-)constant across x and land near the DGP noise level.
    assert implied_sigma.std() < 0.05 * implied_sigma.mean(), (
        f"recovered sigma varies across x (expected one profiled scalar): {implied_sigma}"
    )
    assert 0.5 * noise < implied_sigma.mean() < 2.0 * noise, (
        f"recovered sigma {implied_sigma.mean():.4f} far from DGP noise {noise}"
    )


@pytest.mark.parametrize("formula", _SCAN_FORMULAS)
def test_scan_observation_interval_covers_about_95pct(formula: str) -> None:
    # Held-out coverage: a correct predictive interval covers ~95% of fresh
    # responses drawn from the same DGP.
    noise = 0.25
    train = _make_data(seed=7, n=600, noise=noise)
    model = gamfit.fit(train, formula)
    test = _make_data(seed=8, n=600, noise=noise)
    out = model.predict(
        test[["x"]],
        interval=0.95,
        covariance_mode="conditional",
        observation_interval=True,
        return_type="dict",
    )
    olo = np.asarray(out["observation_lower"], dtype=float)
    ohi = np.asarray(out["observation_upper"], dtype=float)
    y = test["y"].to_numpy(dtype=float)
    covered = np.mean((y >= olo) & (y <= ohi))
    assert 0.90 <= covered <= 0.99, (
        f"observation-interval coverage {covered:.3f} far from nominal 0.95"
    )


def test_scan_confidence_interval_unchanged_when_observation_requested() -> None:
    # Issue #398 invariant: requesting the observation interval must not move the
    # mean point or the mean credible band — it only ADDS the two columns.
    formula = _SCAN_FORMULAS[0]
    model = gamfit.fit(_make_data(), formula)
    base = model.predict(
        _grid(), interval=0.95, covariance_mode="conditional", return_type="dict"
    )
    both = model.predict(
        _grid(),
        interval=0.95,
        covariance_mode="conditional",
        observation_interval=True,
        return_type="dict",
    )
    for key in ("mean", "mean_lower", "mean_upper"):
        np.testing.assert_allclose(
            np.asarray(both[key], dtype=float),
            np.asarray(base[key], dtype=float),
            rtol=0.0,
            atol=1e-12,
            err_msg=f"'{key}' shifted when observation_interval was requested",
        )
