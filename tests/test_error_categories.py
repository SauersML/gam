"""One exception hierarchy, chosen by the engine's ``ErrorCategory``.

Every failure the engine reports carries one category (formula, data,
convergence, not fitted, internal). The Python class is raised under the
category's base, and the CLI exits with the category's code, so both front
ends classify a failure the same way without reading its message.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gamfit._rust")

import gamfit


def _categorical_frame(n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 1.0, n)
    grp = np.array(["north", "south", "east"])[np.arange(n) % 3]
    y = np.sin(4.0 * x) + (grp == "south") + rng.normal(0.0, 0.1, n)
    return pd.DataFrame({"y": y, "x": x, "grp": grp})


@pytest.mark.parametrize(
    ("category", "python_bases"),
    [
        (gamfit.errors.FormulaError, (ValueError,)),
        (gamfit.errors.DataError, (ValueError,)),
        (gamfit.errors.ConvergenceError, (RuntimeError,)),
        (gamfit.errors.NotFittedError, (ValueError, AttributeError)),
        (gamfit.errors.InternalError, (RuntimeError,)),
    ],
)
def test_each_category_base_is_a_gamfit_error_and_its_python_builtin(category, python_bases):
    assert issubclass(category, gamfit.errors.GamfitError)
    for base in python_bases:
        assert issubclass(category, base)


def test_categories_are_disjoint():
    categories = [
        gamfit.errors.FormulaError,
        gamfit.errors.DataError,
        gamfit.errors.ConvergenceError,
        gamfit.errors.NotFittedError,
        gamfit.errors.InternalError,
    ]
    for left in categories:
        for right in categories:
            if left is not right:
                assert not issubclass(left, right), (left, right)


@pytest.mark.parametrize("formula", ["y ~ s(grp)", "y ~ s(x) + s(grp)", "y ~ te(x, grp)", "y ~ linear(grp)"])
def test_a_numeric_term_on_a_string_column_is_a_formula_error_naming_the_column(formula):
    with pytest.raises(gamfit.errors.FormulaError) as caught:
        gamfit.fit(_categorical_frame(), formula)
    message = str(caught.value)
    assert "'grp'" in message
    assert "'north' at row 1" in message
    for alternative in ("factor(grp)", "group(grp)", "s(x, by=grp)", "fs(x, grp)", 's(grp, bs="re")'):
        assert alternative in message, (alternative, message)


def test_a_pandas_categorical_column_is_refused_the_same_way():
    frame = _categorical_frame()
    frame["grp"] = frame["grp"].astype("category")
    with pytest.raises(gamfit.errors.FormulaError, match="'grp'"):
        gamfit.fit(frame, "y ~ s(grp)")


def test_a_stray_string_in_a_numeric_object_column_is_reported_with_its_row():
    frame = _categorical_frame()
    values = frame["x"].astype(object)
    values.iloc[4] = "oops"
    frame["x"] = values
    with pytest.raises(gamfit.errors.FormulaError) as caught:
        gamfit.fit(frame, "y ~ s(x)")
    message = str(caught.value)
    assert "'x'" in message
    assert "'oops' at row 5" in message
    assert "meant to be numeric" in message


@pytest.mark.parametrize("formula", ["y ~ s(x, by=grp)", "y ~ fs(x, grp)", 'y ~ s(grp, bs="re")', "y ~ factor(grp) + s(x)"])
def test_the_terms_the_refusal_suggests_accept_the_same_column(formula):
    model = gamfit.fit(_categorical_frame(), formula)
    assert np.all(np.isfinite(np.asarray(model.predict(_categorical_frame()), dtype=float)))


def test_a_malformed_formula_is_a_formula_error():
    with pytest.raises(gamfit.errors.FormulaError):
        gamfit.fit(_categorical_frame(), "y ~ s(")


def test_a_missing_column_is_a_formula_error():
    with pytest.raises(gamfit.errors.ColumnNotFoundError) as caught:
        gamfit.fit(_categorical_frame(), "y ~ s(absent_column)")
    assert isinstance(caught.value, gamfit.errors.FormulaError)
    assert caught.value.column == "absent_column"


def test_non_finite_data_is_a_data_error():
    frame = _categorical_frame()
    frame.loc[7, "x"] = np.nan
    with pytest.raises(gamfit.errors.DataError, match="'x'"):
        gamfit.fit(frame, "y ~ s(x)")


def test_an_unpredictable_cell_is_a_data_error():
    model = gamfit.fit(_categorical_frame(), "y ~ s(x) + factor(grp)")
    new = _categorical_frame(6)
    new.loc[2, "grp"] = "west"
    with pytest.raises(gamfit.errors.PredictInputError, match="'west'") as caught:
        model.predict(new)
    assert isinstance(caught.value, gamfit.errors.PredictionError)
    assert isinstance(caught.value, gamfit.errors.DataError)


def test_an_unfinished_solve_is_a_convergence_error():
    rng = np.random.RandomState(7)
    n = 30
    t = np.sort(rng.uniform(-1.0, 1.0, n))
    y = np.c_[np.sin(2.0 * t), t**2, np.cos(1.5 * t)]
    # Zero trust-region iterations leave a non-stationary caller start untouched.
    with pytest.raises(gamfit.errors.ConvergenceError) as caught:
        gamfit.reml.gaussian_reml_optimize_latent(
            y=y,
            n_obs=n,
            latent_dim=1,
            centers=np.linspace(-1.0, 1.0, 16).reshape(-1, 1),
            penalty=np.eye(16),
            t=t,
            init="caller",
            max_iter=0,
        )
    assert isinstance(caught.value, gamfit.errors.RemlConvergenceError)
    assert isinstance(caught.value, RuntimeError)


def test_an_unfitted_estimator_raises_not_fitted_error():
    pytest.importorskip("sklearn")
    from gamfit.sklearn import GAMRegressor

    estimator = GAMRegressor(formula="y ~ s(x)")
    with pytest.raises(gamfit.errors.NotFittedError) as caught:
        estimator.predict(_categorical_frame()[["x"]])
    assert isinstance(caught.value, ValueError)
    assert isinstance(caught.value, AttributeError)
    assert "GAMRegressor" in str(caught.value)


def test_python_argument_validation_maps_to_formula_error():
    from gamfit._exceptions import map_exception

    mapped = map_exception(ValueError("bad option"))
    assert type(mapped) is gamfit.errors.FormulaError
    passed = gamfit.errors.DataError("already typed")
    assert map_exception(passed) is passed
