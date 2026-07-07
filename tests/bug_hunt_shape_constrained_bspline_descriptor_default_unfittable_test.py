"""Regression test for #2149: shape-constrained BSpline descriptor defaults.

The formula DSL and the ``smooths={}`` descriptor bridge are documented to lower
an equivalent smooth identically. A shape-constrained ``BSpline`` descriptor used
to emit ``double_penalty=False`` from its Python default, while formula
``s(x, shape=...)`` inherits the Rust/formula default ``double_penalty=True``.
For the shape reparameterization this left null-space directions unpenalized and
collinear with lower-order model columns, causing a pre-fit rank deficiency.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")
pytest.importorskip("gamfit._rust")

import gamfit
from gamfit.smooth import BSpline


def _data(seed: int = 4, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = np.sqrt(x) + rng.normal(0.0, 0.03, n)
    return pd.DataFrame({"x": x, "y": y})


def _grid(df: pd.DataFrame, n: int = 160) -> pd.DataFrame:
    return pd.DataFrame({"x": np.linspace(df["x"].min(), df["x"].max(), n)})


def test_bspline_descriptor_double_penalty_tristate_serialization() -> None:
    default_descriptor = BSpline(shape_constraint="monotone_increasing").to_rust_descriptor()
    assert "double_penalty" not in default_descriptor

    explicit_false = BSpline(double_penalty=False).to_rust_descriptor()
    assert explicit_false["double_penalty"] is False

    explicit_true = BSpline(double_penalty=True).to_rust_descriptor()
    assert explicit_true["double_penalty"] is True


def test_bspline_descriptor_monotone_fits_and_matches_formula() -> None:
    df = _data()
    formula = gamfit.fit(df, "y ~ s(x, shape='monotone_increasing')")
    descriptor = gamfit.fit(
        df,
        "y ~ s(x)",
        smooths={"x": BSpline(shape_constraint="monotone_increasing")},
    )

    pred_formula = np.asarray(formula.predict(_grid(df)), dtype=float)
    pred_descriptor = np.asarray(descriptor.predict(_grid(df)), dtype=float)

    assert np.all(np.diff(pred_descriptor) >= -1e-6)
    np.testing.assert_array_equal(pred_descriptor, pred_formula)


def test_bspline_descriptor_convex_and_concave_fit() -> None:
    df = _data()
    grid = _grid(df)

    convex = gamfit.fit(
        df,
        "y ~ s(x)",
        smooths={"x": BSpline(shape_constraint="convex")},
    )
    convex_pred = np.asarray(convex.predict(grid), dtype=float)
    assert np.all(np.diff(convex_pred, 2) >= -1e-5)

    concave = gamfit.fit(
        df,
        "y ~ s(x)",
        smooths={"x": BSpline(shape_constraint="concave")},
    )
    concave_pred = np.asarray(concave.predict(grid), dtype=float)
    assert np.all(np.diff(concave_pred, 2) <= 1e-5)
