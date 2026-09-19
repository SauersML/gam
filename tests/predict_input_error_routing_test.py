"""Predict-time cell errors raise ``PredictInputError`` naming the cell.

A NaN or infinite covariate, a missing categorical label and an unseen factor
level are problems with the rows handed to ``predict``, not with the fitted
model or its schema, so they raise the documented ``PredictInputError`` (a
``PredictionError``) with the column, the row and the offending level in the
message and a ``help:`` line. Before the fix they surfaced as the bare
``GamfitError`` umbrella, and a numeric-coded ``factor(g)`` level only failed deep
inside the design build without naming its column's training levels.
"""

import numpy as np
import pandas as pd
import pytest

import gamfit
from gamfit.errors import PredictInputError, PredictionError, SchemaMismatchError


def _training_frame():
    rng = np.random.default_rng(0)
    n = 240
    x = rng.uniform(0.0, 1.0, n)
    codes = rng.integers(0, 3, n)
    y = np.sin(6.0 * x) + codes + rng.normal(0.0, 0.3, n)
    labels = np.array([f"L{code}" for code in codes], dtype=object)
    return pd.DataFrame({"x": x, "y": y, "g": labels, "k": codes.astype(float)})


@pytest.fixture(scope="module")
def frame():
    return _training_frame()


@pytest.fixture(scope="module")
def label_model(frame):
    return gamfit.fit(frame, "y ~ s(x) + g")


def _predict_frame(x, g):
    return pd.DataFrame({"x": np.asarray(x, dtype=float), "g": np.array(g, dtype=object)})


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_covariate_raises_predict_input_error(label_model, bad):
    with pytest.raises(PredictInputError) as excinfo:
        label_model.predict(_predict_frame([0.5, bad], ["L0", "L1"]))
    message = str(excinfo.value)
    assert "column 'x'" in message and "row 2" in message, message
    assert "help:" in message, message
    assert isinstance(excinfo.value, PredictionError)
    assert not isinstance(excinfo.value, SchemaMismatchError)


def test_non_finite_covariate_in_a_dict_raises_predict_input_error(label_model):
    with pytest.raises(PredictInputError, match="column 'x'"):
        label_model.predict(
            {"x": np.array([0.5, np.nan]), "g": np.array(["L0", "L1"], dtype=object)}
        )


@pytest.mark.parametrize("formula", ["y ~ s(x) + g", "y ~ s(x) + factor(g)"])
def test_unseen_label_raises_predict_input_error_naming_column_and_level(frame, formula):
    model = gamfit.fit(frame, formula)
    with pytest.raises(PredictInputError) as excinfo:
        model.predict(_predict_frame([0.5, 0.5], ["L0", "LNEW"]))
    message = str(excinfo.value)
    assert "unseen level 'LNEW'" in message and "'g'" in message, message
    assert "L0" in message and "L2" in message, message
    assert "help:" in message, message


def test_missing_label_raises_predict_input_error(label_model):
    with pytest.raises(PredictInputError) as excinfo:
        label_model.predict(_predict_frame([0.5, 0.5], ["L0", None]))
    message = str(excinfo.value)
    assert "'g'" in message and "row 2" in message, message


def test_unseen_numeric_factor_level_raises_predict_input_error(frame):
    model = gamfit.fit(frame, "y ~ s(x) + factor(k)")
    with pytest.raises(PredictInputError) as excinfo:
        model.predict(pd.DataFrame({"x": [0.5, 0.5], "k": [1.0, 7.0]}))
    message = str(excinfo.value)
    assert "unseen level '7'" in message and "'k'" in message, message
    assert "row 2" in message, message


def test_check_reports_the_unseen_numeric_factor_level_with_its_column(frame):
    model = gamfit.fit(frame, "y ~ s(x) + factor(k)")
    check = model.check(pd.DataFrame({"x": [0.5, 0.5], "k": [1.0, 7.0]}))
    assert not check.ok
    assert [issue.column for issue in check.issues] == ["k"], check.issues
    assert "unseen level '7'" in check.issues[0].message


def test_seen_levels_still_predict(label_model):
    predictions = label_model.predict(_predict_frame([0.25, 0.75], ["L0", "L2"]))
    assert np.all(np.isfinite(np.asarray(predictions, dtype=float)))
