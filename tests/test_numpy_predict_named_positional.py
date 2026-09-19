"""NumPy prediction speaks the DataFrame path's column names in both directions.

A NumPy prediction table used to come back as ``np.column_stack`` of the
output columns: six anonymous float columns for an interval request, whose
meaning only the documented column order recorded. It is now a structured
array with one named field per output column, read by the same names as the
DataFrame result.

A positional NumPy input after a fit on a named table used to be labelled
``x0, x1, ...`` and refused with ``SchemaMismatchError`` (the model reads
``a`` and ``z``). The model now binds it to its predictor columns in
training-table order when the width matches, and refuses any other width with
a message naming the expected columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gamfit
from gamfit.errors import SchemaMismatchError


def _table_fit() -> tuple[gamfit.Model, pd.DataFrame]:
    rng = np.random.default_rng(3)
    n = 300
    # The response sits between the predictors and the predictors are not in
    # alphabetical order, so training-table order differs from both the
    # formula's and the sorted order.
    z = rng.uniform(-1.0, 1.0, n)
    a = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * a) + 0.7 * z + rng.normal(0.0, 0.2, n)
    train = pd.DataFrame({"z": z, "y": y, "a": a})
    model = gamfit.fit(train, "y ~ s(a) + z", family="gaussian")
    test = pd.DataFrame({"z": np.linspace(-0.8, 0.8, 9), "a": np.linspace(0.9, 0.1, 9)})
    return model, test


def test_numpy_interval_output_has_the_dataframe_column_names() -> None:
    model, test = _table_fit()
    frame = model.predict(test, interval=0.95, return_type="pandas")

    table = model.predict(test, interval=0.95, return_type="numpy")

    assert isinstance(table, np.ndarray)
    assert table.shape == (len(test),)
    assert table.dtype.names == tuple(frame.columns)
    assert "posterior_mean_lower" in table.dtype.names
    for name in frame.columns:
        np.testing.assert_array_equal(table[name], frame[name].to_numpy(dtype=float))


def test_numpy_predict_after_dataframe_fit_binds_training_order() -> None:
    model, test = _table_fit()
    expected = model.predict(test, interval=0.95, return_type="pandas")

    positional = test[["z", "a"]].to_numpy()
    table = model.predict(positional, interval=0.95)

    # A NumPy input mirrors to a NumPy (structured) result.
    assert table.dtype.names == tuple(expected.columns)
    for name in expected.columns:
        np.testing.assert_allclose(
            table[name], expected[name].to_numpy(dtype=float), rtol=0.0, atol=1e-12
        )
    np.testing.assert_allclose(
        model.predict(positional), model.predict(test), rtol=0.0, atol=1e-12
    )


def test_numpy_predict_with_wrong_width_names_the_expected_columns() -> None:
    model, test = _table_fit()
    wrong = np.column_stack([test["z"], test["a"], test["a"]])

    with pytest.raises(SchemaMismatchError, match=r"\['z', 'a'\]"):
        model.predict(wrong)
