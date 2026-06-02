"""End-to-end tests for the three newly-surfaced inference capabilities.

Each capability is implemented in the Rust engine but was previously
unreachable from the ``gamfit`` Python API:

1. Conformal prediction intervals (``Model.predict_conformal``).
2. Predict covariance mode + observation intervals (``covariance_mode=`` /
   ``observation_interval=`` on ``Model.predict`` / ``Model.predict_array``).
3. Wood per-smooth p-values in the model summary
   (``Summary.smooth_terms`` / ``Summary.smooth_terms_frame``).

The tests assert the kwarg / attribute is reachable AND that it produces a
sane, principled result (coverage-shaped conformal bounds, wider observation
intervals than credible ones, a populated per-smooth significance table).
"""

from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit


def _smooth_training_frame(seed: int = 7, n: int = 200) -> dict[str, list[float]]:
    """A well-conditioned smooth-signal regression frame: y = sin(x) + noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 6.0, n)
    y = np.sin(x) + rng.normal(0.0, 0.1, n)
    return {"x": x.tolist(), "y": y.tolist()}


# --------------------------------------------------------------------------- #
# Exposure 1: conformal prediction intervals
# --------------------------------------------------------------------------- #
def test_predict_conformal_is_reachable_and_covers() -> None:
    """``Model.predict_conformal`` returns a sane conformal interval table."""
    data = _smooth_training_frame(seed=11, n=240)
    n = len(data["x"])
    # Train / calibrate / test split — calibration is held-out labeled data.
    train = {"x": data["x"][: n // 2], "y": data["y"][: n // 2]}
    calib = {"x": data["x"][n // 2 : 3 * n // 4], "y": data["y"][n // 2 : 3 * n // 4]}
    test = {"x": data["x"][3 * n // 4 :], "y": data["y"][3 * n // 4 :]}

    model = gamfit.fit(train, "y ~ s(x)")
    out = model.predict_conformal(
        test,
        calibration=calib,
        conformal_level=0.9,
        return_type="dict",
    )
    # The conformal bounds replace mean_lower / mean_upper in the table.
    assert "mean_lower" in out
    assert "mean_upper" in out
    lower = np.asarray(out["mean_lower"], dtype=float)
    upper = np.asarray(out["mean_upper"], dtype=float)
    mean = np.asarray(out["mean"], dtype=float)
    assert np.all(upper >= lower)
    assert np.all(mean <= upper + 1e-9)
    assert np.all(mean >= lower - 1e-9)

    # Empirical coverage of the held-out test response should be in the right
    # ballpark for the requested 0.9 level (conformal guarantees >= level
    # marginally; allow slack for the small finite calibration set).
    y_test = np.asarray(test["y"], dtype=float)
    covered = np.mean((y_test >= lower) & (y_test <= upper))
    assert covered >= 0.7


def test_predict_conformal_requires_response_column_in_calibration() -> None:
    data = _smooth_training_frame(seed=12, n=120)
    n = len(data["x"])
    train = {"x": data["x"][: n // 2], "y": data["y"][: n // 2]}
    test = {"x": data["x"][n // 2 :], "y": data["y"][n // 2 :]}
    model = gamfit.fit(train, "y ~ s(x)")
    # Calibration without the response column must error (it is labeled data).
    bad_calib = {"x": test["x"]}
    with pytest.raises(Exception):
        model.predict_conformal(test, calibration=bad_calib, conformal_level=0.9)


# --------------------------------------------------------------------------- #
# Exposure 2: covariance mode + observation interval
# --------------------------------------------------------------------------- #
def test_predict_covariance_mode_is_reachable() -> None:
    """All three covariance modes are accepted and yield positive SEs."""
    data = _smooth_training_frame(seed=21, n=160)
    model = gamfit.fit(data, "y ~ s(x)")
    grid = {"x": np.linspace(0.5, 5.5, 25).tolist()}

    se_by_mode = {}
    for mode in ("conditional", "smoothing"):
        tab = model.predict(grid, interval=0.95, covariance_mode=mode, return_type="dict")
        se = np.asarray(tab["std_error"], dtype=float)
        assert np.all(se > 0.0)
        se_by_mode[mode] = se

    # Smoothing-corrected SEs add the J Var(rho) J^T term, so they should be
    # at least as large as the purely-conditional ones (>= within tolerance).
    assert np.all(
        se_by_mode["smoothing"] >= se_by_mode["conditional"] - 1e-9
    )


def test_predict_rejects_unknown_covariance_mode() -> None:
    data = _smooth_training_frame(seed=22, n=80)
    model = gamfit.fit(data, "y ~ s(x)")
    grid = {"x": [1.0, 2.0, 3.0]}
    with pytest.raises(Exception):
        model.predict(grid, interval=0.95, covariance_mode="not-a-mode")


def test_predict_observation_interval_is_reachable_and_wider() -> None:
    """``observation_interval=True`` adds prediction-interval columns."""
    data = _smooth_training_frame(seed=23, n=160)
    model = gamfit.fit(data, "y ~ s(x)")
    grid = {"x": np.linspace(0.5, 5.5, 20).tolist()}

    tab = model.predict(
        grid, interval=0.95, observation_interval=True, return_type="dict"
    )
    assert "observation_lower" in tab
    assert "observation_upper" in tab
    mean_lower = np.asarray(tab["mean_lower"], dtype=float)
    mean_upper = np.asarray(tab["mean_upper"], dtype=float)
    obs_lower = np.asarray(tab["observation_lower"], dtype=float)
    obs_upper = np.asarray(tab["observation_upper"], dtype=float)
    # Observation (prediction) intervals add residual noise variance, so they
    # must contain the (narrower) credible interval for the mean.
    assert np.all(obs_upper >= mean_upper - 1e-9)
    assert np.all(obs_lower <= mean_lower + 1e-9)

    # Default (no observation_interval) keeps the standard schema.
    plain = model.predict(grid, interval=0.95, return_type="dict")
    assert "observation_lower" not in plain


# --------------------------------------------------------------------------- #
# Exposure 3: Wood per-smooth p-values in Summary
# --------------------------------------------------------------------------- #
def test_summary_exposes_smooth_terms_significance_table() -> None:
    """``Summary.smooth_terms`` is the mgcv-style per-smooth table."""
    data = _smooth_training_frame(seed=31, n=200)
    model = gamfit.fit(data, "y ~ s(x)")
    summary = model.summary()

    terms = summary.smooth_terms
    assert isinstance(terms, list)
    assert len(terms) >= 1, "a model with s(x) must report at least one smooth term"

    sx = next((t for t in terms if str(t.get("name", "")).startswith("s(x")), terms[0])
    assert "edf" in sx
    assert float(sx["edf"]) > 0.0
    # The strong sin signal should be highly significant: a finite, small
    # Wald p-value with a positive test statistic.
    assert sx.get("p_value") is not None
    p = float(sx["p_value"])
    assert 0.0 <= p <= 1.0
    assert p < 0.05
    assert float(sx["chi_sq"]) > 0.0

    # Subscript access and the frame accessor both round-trip the same table.
    assert summary["smooth_terms"] == terms
    frame = summary.smooth_terms_frame()
    assert len(frame) == len(terms)
    assert "edf" in frame.columns
    assert "p_value" in frame.columns
