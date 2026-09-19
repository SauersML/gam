"""Bug hunt: a missing value in a column the formula never references makes
``gamfit.fit`` / ``predict`` / ``check`` / ``validate_formula`` refuse the whole
table, even though every column the model actually uses is complete.

Real-data repro (R ``datasets::airquality``, 153 rows): ``Ozone`` has 37 NAs and
``Solar.R`` has 7, while ``Wind`` and ``Temp`` are complete. The model
``Temp ~ s(Wind)`` touches neither incomplete column, yet

    gamfit.fit(airquality_df, "Temp ~ s(Wind)")
    -> GamfitError: null value at row 5, column 'Ozone'

Projecting the frame down to the two modelled columns by hand fits all 153 rows
(deviance 10763.32). The documented remedy in ``docs/data-input.md`` ("Drop or
impute rows upstream: ``df.dropna(subset=[...])``") does not help, because
``dropna(subset=["Temp", "Wind"])`` removes nothing here — the NAs are in the
columns the model does not use. The only remedies are ``df.dropna()``, which
throws away 42 of 153 rows and changes the answer (deviance 7151.12), or
hand-projecting the columns, which the docs never mention.

The Rust CLI reads the same CSV with the NA cells intact and fits all 153 rows,
so ``gam fit airquality.csv "Temp ~ s(Wind)"`` and
``gamfit.fit(pandas.read_csv("airquality.csv"), "Temp ~ s(Wind)")`` disagree on
whether the data is usable at all.

Mechanism: the requirement is enforced during generic table INGESTION, before
the formula is known, so it cannot be scoped to the modelled columns.
``crates/gam-pyffi/src/model/model_ffi.rs`` ``encoded_table_from_columns``
(~line 786) walks every numeric column and returns on the first non-finite cell
(~lines 833-840); the Arrow path does the same via
``gam_data::reject_arrow_null_values`` (``crates/gam-data/src/lib.rs``, ~line
1416), and ``stringify_cell`` rejects ``None`` / empty strings in unreferenced
categorical columns. A correctly scoped, role-aware validator already exists in
``crates/gam-models/src/fit_orchestration/materialize/columns.rs`` (~line 18),
which reports ``"{role} column '{name}' contains non-finite value at row {row}"``
only for columns a term actually consumes.

Observed: any missing cell anywhere in the frame aborts fit / predict / check /
validate_formula.

Expected: the completeness requirement applies to the columns the formula
references (response, covariates, ``by=``, weights, offset, id). A frame that
carries additional columns with missing values fits exactly as the projected
frame does.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def _clean_data() -> dict[str, Any]:
    """A well-conditioned univariate smoothing problem with complete columns."""
    rng = np.random.default_rng(20260823)
    x = np.linspace(0.0, 1.0, 120)
    y = np.sin(2.0 * np.pi * x) + 0.15 * rng.standard_normal(x.size)
    return {"x": x, "y": y}


def _reference_fit() -> tuple[Any, np.ndarray, np.ndarray]:
    data = _clean_data()
    model = gamfit.fit(data, "y ~ s(x)", family="gaussian")
    grid = np.linspace(0.05, 0.95, 11)
    preds = np.asarray(
        model.predict({"x": grid}, return_type="dict")["posterior_mean"], dtype=float
    )
    return model, grid, preds


def _with_unreferenced_gap(missing: Any, dtype: Any = float) -> dict[str, Any]:
    """The same problem plus one column the formula never mentions, holed at row 7."""
    data = _clean_data()
    column = np.full(data["x"].size, 1.0 if dtype is float else "a", dtype=dtype)
    column[7] = missing
    return {**data, "unused": column}


@pytest.mark.parametrize(
    ("label", "missing", "dtype"),
    [
        ("numeric_nan", np.nan, float),
        ("numeric_inf", np.inf, float),
        ("categorical_empty", "", object),
        ("categorical_none", None, object),
    ],
)
def test_fit_ignores_missing_values_in_unreferenced_columns(
    label: str, missing: Any, dtype: Any
) -> None:
    """A hole in a column no term consumes must not change (or block) the fit."""
    _, grid, expected = _reference_fit()
    reference = gamfit.fit(_clean_data(), "y ~ s(x)", family="gaussian").summary()

    model = gamfit.fit(_with_unreferenced_gap(missing, dtype), "y ~ s(x)", family="gaussian")
    summary = model.summary()

    assert summary.n_obs == reference.n_obs, (
        f"[{label}] the unreferenced column changed the fitted sample size: "
        f"{summary.n_obs} vs {reference.n_obs}"
    )
    assert summary.deviance == pytest.approx(reference.deviance, rel=1e-10), (
        f"[{label}] the unreferenced column changed the deviance: "
        f"{summary.deviance} vs {reference.deviance}"
    )
    got = np.asarray(
        model.predict({"x": grid}, return_type="dict")["posterior_mean"], dtype=float
    )
    np.testing.assert_allclose(
        got,
        expected,
        rtol=1e-10,
        atol=1e-10,
        err_msg=f"[{label}] predictions changed because of an unreferenced column",
    )


def test_an_all_missing_unreferenced_column_fits_bitwise_identically() -> None:
    """The fit's input contract is the columns it reads; an extra all-NaN column is not among them."""
    reference, grid, expected = _reference_fit()
    padded = {**_clean_data(), "unused": np.full(_clean_data()["x"].size, np.nan)}
    model = gamfit.fit(padded, "y ~ s(x)", family="gaussian")

    assert model.summary().deviance == reference.summary().deviance
    got = np.asarray(
        model.predict({"x": grid}, return_type="dict")["posterior_mean"], dtype=float
    )
    np.testing.assert_array_equal(got, expected)


def test_a_missing_value_in_a_used_column_is_still_refused_by_name() -> None:
    """The contract narrows what is validated, never what a used column may hold."""
    data = _clean_data()
    data["x"] = data["x"].copy()
    data["x"][7] = np.nan
    with pytest.raises(Exception, match="column 'x' has non-finite value NaN at row 8"):
        gamfit.fit(data, "y ~ s(x)", family="gaussian")


def test_predict_ignores_missing_values_in_unreferenced_columns() -> None:
    """Serving frames routinely carry extra, partly-missing columns."""
    model, grid, expected = _reference_fit()

    padded: dict[str, Any] = {"x": grid, "unused": np.full(grid.size, np.nan)}
    got = np.asarray(
        model.predict(padded, return_type="dict")["posterior_mean"], dtype=float
    )

    np.testing.assert_allclose(
        got,
        expected,
        rtol=1e-10,
        atol=1e-10,
        err_msg="predict rejected or altered a frame over an unreferenced column",
    )


def test_check_accepts_a_frame_whose_unreferenced_column_has_a_gap() -> None:
    """``check()`` is the documented guard; it must answer, not raise."""
    model, grid, _ = _reference_fit()

    padded: dict[str, Any] = {"x": grid, "unused": np.full(grid.size, np.nan)}
    result = model.check(padded)

    assert result.ok, (
        "check() reported issues for a frame whose modelled columns are complete: "
        f"{[(issue.kind, issue.column) for issue in (result.issues or [])]}"
    )


def test_validate_formula_ignores_missing_values_in_unreferenced_columns() -> None:
    validation = gamfit.validate_formula(
        _with_unreferenced_gap(np.nan), "y ~ s(x)", family="gaussian"
    )
    assert validation["response_column"] == "y"


def test_arrow_nulls_in_unreferenced_columns_obey_the_same_projection_contract() -> None:
    """Arrow nulls are absence, not a reason to validate an unused column."""
    pa = pytest.importorskip("pyarrow")
    clean = _clean_data()
    rows = clean["x"].size
    unused = pa.array(
        [None if row == 7 else 1.0 for row in range(rows)],
        type=pa.float64(),
    )
    training = pa.table({"x": clean["x"], "y": clean["y"], "unused": unused})

    model = gamfit.fit(training, "y ~ s(x)", family="gaussian")
    reference = gamfit.fit(_clean_data(), "y ~ s(x)", family="gaussian")
    assert model.summary().n_obs == reference.summary().n_obs
    assert model.summary().deviance == pytest.approx(
        reference.summary().deviance, rel=1e-10
    )

    grid = np.linspace(0.05, 0.95, 11)
    serving = pa.table(
        {
            "x": grid,
            "unused": pa.array(
                [None if row == 4 else "" for row in range(grid.size)],
                type=pa.string(),
            ),
        }
    )
    assert model.check(serving).ok
    got = np.asarray(
        model.predict(serving, return_type="dict")["posterior_mean"], dtype=float
    )
    expected = np.asarray(
        reference.predict({"x": grid}, return_type="dict")["posterior_mean"],
        dtype=float,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)

    validation = gamfit.validate_formula(
        training, "y ~ s(x)", family="gaussian"
    )
    assert validation["response_column"] == "y"
