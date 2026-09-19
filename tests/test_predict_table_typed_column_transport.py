"""``predict_table`` hands columns to Python as float64 arrays, not JSON text.

The prediction columns used to cross the FFI as one JSON document that Python
decoded, re-encoded for a Rust column-ordering call, decoded again, and then
walked value by value into arrays -- at a million prediction rows that round
trip cost more than the prediction itself. The binding now returns a dict
whose ``columns`` map each name, already in the Rust preferred order, to a
float64 ``ndarray`` built from the Rust vector, so ``Model.predict`` never
re-encodes the payload.
"""

from __future__ import annotations

import numpy as np

import gamfit
from gamfit._binding import rust_module
from gamfit._tables import normalize_table


def _fitted() -> tuple[gamfit.Model, dict[str, np.ndarray]]:
    rng = np.random.default_rng(7)
    n = 400
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    eta = np.sin(2.0 * x) + 0.5 * z
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    model = gamfit.fit({"x": x, "z": z, "y": y}, "y ~ s(x) + z", family="binomial")
    grid = {"x": np.linspace(-0.9, 0.9, 25), "z": np.linspace(0.5, -0.5, 25)}
    return model, grid


def test_predict_table_returns_ordered_float64_array_columns() -> None:
    model, grid = _fitted()
    headers, rows, _ = normalize_table(grid)

    payload = rust_module().predict_table(
        model._prediction_model, headers, rows, 0.95, None, False
    )

    assert isinstance(payload, dict)
    assert payload["point_column"] == "posterior_mean"
    assert payload["point_shape"] == "estimand_explicit"
    columns = payload["columns"]
    assert list(columns) == [
        "linear_predictor_plugin",
        "mean_plugin",
        "posterior_mean",
        "posterior_mean_standard_error",
        "posterior_mean_lower",
        "posterior_mean_upper",
    ]
    for values in columns.values():
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.float64
        assert values.shape == (25,)

    table = model.predict(grid, interval=0.95, return_type="dict")
    for name, values in columns.items():
        np.testing.assert_array_equal(np.asarray(table[name]), values)
    np.testing.assert_array_equal(model.predict(grid), columns["posterior_mean"])
    assert table["covariance_source"] == payload["covariance_source"]
