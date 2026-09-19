"""Summary smooth-term p-values under prior weights and a location-scale fit.

* For a fixed-dispersion family a prior weight ``w`` and ``w`` literal copies of
  a row enter the likelihood identically, so the fitted model -- and every
  number of its summary table -- must be the same. The spline's sum-to-zero
  centering is taken over the row multiset, so the two fits carry the same
  smooth in different centering gauges; the Wald statistic must not depend on
  that gauge.
* A Gaussian location-scale fit (``noise_formula``) is the model that does
  describe a heteroscedastic response; its mean smooths must get a summary
  table, not a stale-layout refusal.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_FORMULA = "y ~ s(x1) + s(x2)"


def _rows(model: Any) -> dict[str, dict[str, float]]:
    return {row["name"]: row for row in model.summary().smooth_terms}


def _assert_same_table(weighted: Any, duplicated: Any) -> None:
    a, b = _rows(weighted), _rows(duplicated)
    assert set(a) == set(b) == {"s(x1)", "s(x2)"}, (list(a), list(b))
    for name in a:
        for key in ("edf", "ref_df", "chi_sq", "p_value"):
            assert a[name][key] == pytest.approx(b[name][key], rel=1e-5, abs=1e-12), (
                name,
                key,
                a[name][key],
                b[name][key],
            )


def _covariates(rng: Any, n: int) -> tuple[Any, Any]:
    return rng.uniform(size=n), rng.uniform(size=n)


def test_poisson_integer_weights_match_duplicated_rows() -> None:
    rng = np.random.default_rng(3031)
    n = 150
    x1, x2 = _covariates(rng, n)
    w = rng.integers(1, 4, size=n)
    eta = 0.3 + 0.5 * np.sin(2 * np.pi * x1) + 0.4 * np.cos(2 * np.pi * x2)
    y = rng.poisson(np.exp(eta)).astype(float)
    idx = np.repeat(np.arange(n), w)
    weighted = gamfit.fit(
        {"y": y, "x1": x1, "x2": x2, "w": w.astype(float)},
        _FORMULA,
        family="poisson",
        weights="w",
    )
    duplicated = gamfit.fit({"y": y[idx], "x1": x1[idx], "x2": x2[idx]}, _FORMULA, family="poisson")
    _assert_same_table(weighted, duplicated)


def test_binomial_trials_match_expanded_bernoulli_rows() -> None:
    rng = np.random.default_rng(3032)
    n = 150
    x1, x2 = _covariates(rng, n)
    trials = rng.integers(1, 6, size=n)
    prob = 1.0 / (1.0 + np.exp(-(np.sin(2 * np.pi * x1) + 0.8 * np.cos(2 * np.pi * x2))))
    successes = rng.binomial(trials, prob)
    proportion = gamfit.fit(
        {"y": successes / trials, "x1": x1, "x2": x2, "m": trials.astype(float)},
        _FORMULA,
        family="binomial",
        weights="m",
    )
    idx = np.repeat(np.arange(n), trials)
    bernoulli = np.concatenate([np.r_[np.ones(s), np.zeros(t - s)] for s, t in zip(successes, trials)])
    expanded = gamfit.fit({"y": bernoulli, "x1": x1[idx], "x2": x2[idx]}, _FORMULA, family="binomial")
    _assert_same_table(proportion, expanded)


def test_a_location_scale_fit_tabulates_its_mean_smooths() -> None:
    rng = np.random.default_rng(3033)
    n = 400
    x1, x2 = _covariates(rng, n)
    y = np.sin(2 * np.pi * x1) + np.exp(-1.0 + 2.0 * x2) * rng.standard_normal(n)
    summary = gamfit.fit(
        {"y": y, "x1": x1, "x2": x2}, _FORMULA, family="gaussian", noise_formula="s(x2)"
    ).summary()
    assert summary.smooth_terms_unavailable is None, summary.smooth_terms_unavailable
    rows = {row["name"]: row for row in summary.smooth_terms}
    assert {"s(x1)", "s(x2)"} <= set(rows), list(rows)
    for name in ("s(x1)", "s(x2)"):
        p = rows[name]["p_value"]
        assert p is not None and 0.0 <= p <= 1.0, (name, p)
    # The mean truly depends on x1 and not on x2.
    assert rows["s(x1)"]["p_value"] < 1e-6, rows["s(x1)"]
    # No scalar family, so no AIC -- stated, not failed.
    assert summary.aic_corrected is None
    assert summary.aic_corrected_unavailable, summary
