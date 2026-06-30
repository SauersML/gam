"""Smoke tests for the multinomial-logit inference surface (#1101).

Before #1101 the multinomial family was the single regression family entirely
absent from the unified predict/inference surface: no prediction intervals, no
``std_error``, no posterior-predictive draws, and a ``summary()`` with no
smooth-term p-values. These tests assert that each of those paths now exists and
returns sane (finite, correctly shaped, simplex-consistent) values for a
REML-fitted multinomial GAM with smooth terms.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _softmax_dataset(seed: int, n: int = 600):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.0, 2.0, n)
    x2 = rng.uniform(-2.0, 2.0, n)
    eta = np.stack(
        [
            np.zeros_like(x1),
            0.8 * x1 - 0.5 * x2,
            -0.6 * x1 + 0.7 * x2 + 0.3,
        ],
        axis=1,
    )
    probs = np.exp(eta)
    probs /= probs.sum(axis=1, keepdims=True)
    cls = np.array([rng.choice(3, p=row) for row in probs])
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": cls.astype(str)})
    return df


def _fit():
    df = _softmax_dataset(seed=20260630)
    model = gamfit.fit(df, "y ~ s(x1) + s(x2)", family="multinomial")
    return model, df


def test_confidence_interval_returns_sane_bounds() -> None:
    model, df = _fit()
    pred = model.predict(df, interval="confidence", level=0.95)
    k = len(model.classes_)
    n = len(df)
    assert pred.mean.shape == (n, k)
    assert pred.std_error.shape == (n, k)
    assert pred.mean_lower.shape == (n, k)
    assert pred.mean_upper.shape == (n, k)
    se = np.asarray(pred.std_error, dtype=float)
    lo = np.asarray(pred.mean_lower, dtype=float)
    hi = np.asarray(pred.mean_upper, dtype=float)
    mean = np.asarray(pred.mean, dtype=float)
    assert np.all(np.isfinite(se)) and np.all(se >= 0.0)
    # Simplex-clamped band brackets the point estimate within [0, 1].
    assert np.all(lo >= -1e-9) and np.all(hi <= 1.0 + 1e-9)
    assert np.all(lo <= mean + 1e-9) and np.all(hi >= mean - 1e-9)
    # Mean rows still sum to 1.
    assert np.allclose(mean.sum(axis=1), 1.0, atol=1e-6)


def test_std_error_alias_matches_interval() -> None:
    model, df = _fit()
    se_direct = np.asarray(model.std_error(df), dtype=float)
    se_interval = np.asarray(
        model.predict(df, interval="confidence").std_error, dtype=float
    )
    assert np.allclose(se_direct, se_interval)


def test_smooth_significance_table_is_populated() -> None:
    model, _ = _fit()
    sig = model.smooth_significance()
    assert isinstance(sig, list)
    assert len(sig) > 0
    for row in sig:
        for key in ("class", "term", "edf", "ref_df", "statistic", "p_value"):
            assert key in row
        assert np.isfinite(row["edf"]) and row["edf"] > 0.0
        assert np.isfinite(row["statistic"]) and row["statistic"] >= 0.0
        assert 0.0 <= row["p_value"] <= 1.0
    # The summary embeds the same table.
    text = model.summary()
    assert "smooth terms" in text


def test_posterior_predict_draws_sane_labels() -> None:
    model, df = _fit()
    draws = model.posterior_predict(df, n_draws=50, seed=7)
    assert draws.shape == (50, len(df))
    valid = set(model.classes_)
    assert set(np.unique(draws)).issubset(valid)
    # Deterministic in the seed.
    draws2 = model.posterior_predict(df, n_draws=50, seed=7)
    assert np.array_equal(draws, draws2)
    # A different seed changes the draw stream.
    draws3 = model.posterior_predict(df, n_draws=50, seed=8)
    assert not np.array_equal(draws, draws3)
    # Replicate class frequencies track the fitted mean probabilities.
    probs = np.asarray(model.predict(df), dtype=float)
    expected_freq = probs.mean(axis=0)
    for c, name in enumerate(model.classes_):
        emp = float(np.mean(draws == name))
        assert abs(emp - expected_freq[c]) < 0.05
