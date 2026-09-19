"""pyGAM oracle: input validation (pygam/tests/test_utils.py, test_GAM_methods.py).

pyGAM rejects non-finite inputs, out-of-domain responses, mismatched lengths
and negative weights, and extrapolates a spline linearly. The same contracts
are checked here on seeded synthetic data. The last test is the guard the
lane requires: no oracle test may import pyGAM.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import gamfit


def _data(seed: int = 60, n: int = 200) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(6.0 * x) + rng.normal(0.0, 0.3, n)
    return {"x": x, "y": y}


@pytest.fixture(scope="module")
def fitted() -> tuple[dict[str, np.ndarray], gamfit.Model]:
    d = _data()
    return d, gamfit.fit(d, "y ~ s(x)")


# test_utils::test_check_X_not_finite / test_check_y_not_finite
@pytest.mark.parametrize("column", ["x", "y"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_training_value_is_rejected(column: str, bad: float) -> None:
    d = _data()
    d[column] = d[column].copy()
    d[column][17] = bad
    with pytest.raises(Exception):
        gamfit.fit(d, "y ~ s(x)")


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_prediction_input_is_rejected(
    fitted: tuple[dict[str, np.ndarray], gamfit.Model], bad: float
) -> None:
    _, m = fitted
    with pytest.raises(Exception):
        m.predict({"x": np.array([0.2, bad, 0.4])})


# test_utils::test_check_y_* : responses outside the family's support.
@pytest.mark.parametrize(
    ("family", "bad_value"),
    [("binomial", 2.0), ("binomial", -1.0), ("poisson", -1.0), ("gamma", 0.0), ("gamma", -2.0)],
)
def test_out_of_domain_response_is_rejected(family: str, bad_value: float) -> None:
    rng = np.random.default_rng(61)
    n = 200
    x = rng.uniform(0.0, 1.0, n)
    if family == "binomial":
        y = (rng.uniform(size=n) < 0.5).astype(float)
    elif family == "poisson":
        y = rng.poisson(3.0, n).astype(float)
    else:
        y = rng.gamma(2.0, 1.0, n)
    y[5] = bad_value
    with pytest.raises(Exception):
        gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family=family)


# test_utils::test_check_X_y_different_lengths
def test_column_length_mismatch_is_rejected() -> None:
    d = _data()
    with pytest.raises(Exception):
        gamfit.fit({"x": d["x"], "y": d["y"][:-1]}, "y ~ s(x)")
    with pytest.raises(ValueError):
        gamfit.fit_array(d["x"].reshape(-1, 1), d["y"][:-1], "y ~ s(x0)")


# test_GAM_methods::test_input_data_after_fitting (weights)
def test_negative_weight_is_rejected() -> None:
    d = _data()
    d["w"] = np.ones_like(d["x"])
    d["w"][3] = -1.0
    with pytest.raises(Exception):
        gamfit.fit(d, "y ~ s(x)", weights="w")


def test_empty_right_hand_side_is_rejected() -> None:
    with pytest.raises(Exception):
        gamfit.fit(_data(), "y ~ ")


def test_missing_predictor_column_at_predict_is_rejected(
    fitted: tuple[dict[str, np.ndarray], gamfit.Model],
) -> None:
    _, m = fitted
    with pytest.raises(Exception, match="x"):
        m.predict({"z": np.array([0.5])})


# test_GAM_methods::test_extrapolation: beyond the knots the spline is linear.
def test_extrapolation_is_finite_and_linear(
    fitted: tuple[dict[str, np.ndarray], gamfit.Model],
) -> None:
    d, m = fitted
    lo, hi = float(d["x"].min()), float(d["x"].max())
    for grid in (np.linspace(hi + 0.1, hi + 2.0, 12), np.linspace(lo - 2.0, lo - 0.1, 12)):
        curve = np.asarray(m.predict({"x": grid}, return_type="dict")["mean_plugin"], float)
        assert np.all(np.isfinite(curve))
        second = np.diff(curve, n=2)
        assert np.max(np.abs(second)) <= 1e-8 * max(1.0, float(np.max(np.abs(curve))))
        # A non-trivial slope: linear, not a constant clamp.
        assert abs(curve[-1] - curve[0]) > 1e-6


# test_utils::test_check_X_numpy2d / fit on an (n, 1) array.
def test_fit_array_on_column_matrix_matches_table_fit() -> None:
    d = _data()
    ma = gamfit.fit_array(d["x"].reshape(-1, 1), d["y"], "y ~ s(x0)")
    mt = gamfit.fit({"x0": d["x"], "y": d["y"]}, "y ~ s(x0)")
    grid = np.linspace(0.05, 0.95, 20)
    np.testing.assert_allclose(
        np.asarray(ma.predict_array(grid.reshape(-1, 1)), float),
        np.asarray(mt.predict({"x0": grid}), float),
        rtol=1e-10,
        atol=1e-12,
    )


def test_pandas_frame_matches_dict_input() -> None:
    import pandas as pd

    d = _data()
    frame = pd.DataFrame(d)
    np.testing.assert_array_equal(
        np.asarray(gamfit.fit(frame, "y ~ s(x)").predict(frame), float),
        np.asarray(gamfit.fit(d, "y ~ s(x)").predict(d), float),
    )


def test_oracle_suite_never_imports_pygam() -> None:
    """The oracle tests carry recorded expectations; the reference library must
    never be a test-time dependency."""
    pattern = re.compile(r"^\s*(import\s+py" r"gam\b|from\s+py" r"gam\b)", re.MULTILINE)
    files = sorted(Path(__file__).parent.glob("pygam_oracle_*"))
    assert len(files) >= 5, files
    offenders = [f.name for f in files if f.is_file() and pattern.search(f.read_text())]
    assert offenders == []
