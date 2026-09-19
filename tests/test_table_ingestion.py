"""Table ingestion across input libraries.

numpy is gamfit's only dependency, so a pandas frame must fit without pyarrow;
polars frames cross through their own Arrow C stream; and every untyped input
(dict, records, rows, object columns) shares one inference rule and one error
type, ``gamfit.DataError``, naming the column and row of a bad value.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import polars as pl
import pytest

import gamfit
from gamfit._tables import CATEGORICAL_CELL_SENTINEL, normalize_table

# Makes `import pyarrow` fail the way it does when pyarrow is not installed.
_BLOCK_PYARROW = textwrap.dedent(
    """
    import sys

    class _NoPyArrow:
        def find_spec(self, name, path=None, target=None):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    sys.meta_path.insert(0, _NoPyArrow())
    """
)


def _frame_columns(n: int = 120) -> dict[str, list]:
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 1.0, n)
    site = np.array(["north", "south", "east"])[np.arange(n) % 3]
    y = np.sin(5.0 * x) + (site == "south") * 0.5 + rng.normal(0.0, 0.1, n)
    return {"x": x.tolist(), "site": site.tolist(), "y": y.tolist()}


def _rendered(data) -> tuple[list[str], list[list[str]]]:
    headers, rows, _ = normalize_table(data)
    return headers, [list(rows[index]) for index in range(len(rows))]


def test_pandas_frame_fits_and_predicts_without_pyarrow() -> None:
    columns = _frame_columns()
    script = _BLOCK_PYARROW + textwrap.dedent(
        f"""
        import json
        import pandas as pd
        import gamfit

        columns = json.loads({json.dumps(json.dumps(columns))})
        frame = pd.DataFrame(columns)
        frame["site_cat"] = frame["site"].astype("category")
        frame["count"] = pd.array(range(len(frame)), dtype="Int64")
        frame["label"] = frame["site"].astype("string")
        model = gamfit.fit(frame, "y ~ s(x) + factor(site_cat) + factor(label) + count")
        prediction = model.predict(frame)
        assert "pyarrow" not in sys.modules
        print(json.dumps([float(value) for value in prediction]))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    without_pyarrow = np.array(json.loads(completed.stdout))

    frame = dict(columns)
    frame["site_cat"] = columns["site"]
    frame["count"] = [float(index) for index in range(len(columns["x"]))]
    frame["label"] = columns["site"]
    reference = gamfit.fit(frame, "y ~ s(x) + factor(site_cat) + factor(label) + count")
    np.testing.assert_allclose(without_pyarrow, reference.predict(frame), rtol=1e-8)


def test_pandas_categorical_keeps_declared_levels_even_when_numeric() -> None:
    frame = pd.DataFrame(
        {
            "g": pd.Categorical([10, 2, 10, 2], categories=[2, 10, 99]),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    headers, rows = _rendered(frame)
    g = [row[headers.index("g")] for row in rows]
    assert g == [CATEGORICAL_CELL_SENTINEL + level for level in ("10", "2", "10", "2")]


def test_pandas_string_object_and_nullable_columns_match_their_dict_form() -> None:
    frame = pd.DataFrame(
        {
            "s": pd.array(["a", "b", "a"], dtype="string"),
            "o": pd.Series(["0", "1", "0"], dtype=object),
            "n": pd.array([1, 0, 1], dtype="Int64"),
            "b": pd.array([True, False, True], dtype="boolean"),
        }
    )
    as_dict = {
        "s": ["a", "b", "a"],
        "o": ["0", "1", "0"],
        "n": [1, 0, 1],
        "b": [True, False, True],
    }
    assert _rendered(frame) == _rendered(as_dict)


def test_pandas_missing_cells_are_preserved_until_a_term_consumes_them() -> None:
    frame = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x": [0.1, 0.4, 0.2, 0.9, 0.5, 0.7],
            "unused_int": pd.array([1, None, 3, 4, 5, 6], dtype="Int64"),
            "unused_str": pd.array(["a", None, "b", "a", "b", "a"], dtype="string"),
            "unused_cat": pd.Categorical(["a", None, "b", "a", "b", "a"]),
        }
    )
    model = gamfit.fit(frame, "y ~ x")
    assert np.all(np.isfinite(model.predict(frame)))
    with pytest.raises(gamfit.DataError, match="row 2"):
        gamfit.fit(frame, "y ~ x + unused_int")


def test_polars_frame_with_string_categorical_and_enum_columns_fits() -> None:
    columns = _frame_columns()
    frame = pl.DataFrame(columns).with_columns(
        pl.col("site").cast(pl.Categorical).alias("site_cat"),
        pl.col("site").cast(pl.Enum(["east", "north", "south"])).alias("site_enum"),
    )
    formula = "y ~ s(x) + factor(site)"
    reference = gamfit.fit(columns, formula).predict(columns)
    for factor in ("site", "site_cat", "site_enum"):
        model = gamfit.fit(frame, formula.replace("site", factor))
        np.testing.assert_allclose(np.asarray(model.predict(frame)).ravel(), reference, rtol=1e-8)


def test_polars_numeric_string_labels_stay_categorical() -> None:
    frame = pl.DataFrame({"g": ["0", "1", "2"], "y": [1.0, 2.0, 3.0]})
    headers, rows = _rendered(frame)
    assert all(row[headers.index("g")].startswith(CATEGORICAL_CELL_SENTINEL) for row in rows)


@pytest.mark.parametrize(
    ("label", "table", "type_name", "row"),
    [
        ("dict datetime", {"g": [1.0, datetime.datetime(2020, 1, 1)]}, "datetime.datetime", 2),
        ("dict object", {"g": [object(), 1.0]}, "object", 1),
        ("dict complex", {"g": [1.0, 2.0, 1 + 2j]}, "complex", 3),
        ("records", [{"g": 1.0}, {"g": datetime.date(2020, 1, 1)}], "datetime.date", 2),
        (
            "pandas object",
            pd.DataFrame({"g": pd.Series([1.0, datetime.date(2020, 1, 1)], dtype=object)}),
            "datetime.date",
            2,
        ),
    ],
)
def test_unsupported_values_raise_data_error_naming_column_and_row(
    label: str, table, type_name: str, row: int
) -> None:
    with pytest.raises(gamfit.DataError) as caught:
        normalize_table(table)
    message = str(caught.value)
    assert f"at row {row}, column 'g'" in message, message
    assert f"'{type_name}'" in message, message


@pytest.mark.parametrize(
    "table",
    [
        # Nanosecond datetimes cast to integers, so they once fit as numbers.
        {"g": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")},
        {"g": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")},
        {"g": np.array([1, 2], dtype="timedelta64[s]")},
        {"g": np.array([1 + 2j, 3.0])},
        pd.DataFrame({"g": pd.to_datetime(["2020-01-01", "2020-01-02"])}),
        pd.DataFrame({"g": pd.to_timedelta([1, 2], unit="s")}),
    ],
    ids=["numpy-ns", "numpy-day", "numpy-timedelta", "numpy-complex", "pandas-datetime", "pandas-timedelta"],
)
def test_a_declared_dtype_that_is_not_numbers_or_labels_is_a_data_error(table) -> None:
    with pytest.raises(gamfit.DataError, match="unsupported column dtype .* for column 'g'"):
        normalize_table(table)


@pytest.mark.parametrize(
    "table",
    [
        {"g": ["a", "  ", "b"]},
        pd.DataFrame({"g": ["a", "  ", "b"]}),
        pd.DataFrame({"g": pd.Categorical(["a", "  ", "b"])}),
        pl.DataFrame({"g": ["a", "  ", "b"]}),
    ],
    ids=["dict", "pandas", "pandas-categorical", "polars"],
)
def test_a_whitespace_label_is_the_same_data_error_from_every_library(table) -> None:
    with pytest.raises(gamfit.DataError, match="row 2, column 'g'"):
        normalize_table(table)


def test_padded_labels_are_the_same_levels_from_every_library() -> None:
    labels = [" a", "b ", " a "]
    expected = _rendered({"g": labels})
    assert expected[1] == [[CATEGORICAL_CELL_SENTINEL + level] for level in ("a", "b", "a")]
    for table in (
        pd.DataFrame({"g": labels}),
        pd.DataFrame({"g": pd.Categorical(labels)}),
        pl.DataFrame({"g": labels}),
        pl.DataFrame({"g": labels}).with_columns(pl.col("g").cast(pl.Categorical)),
    ):
        assert _rendered(table) == expected, type(table)


def test_unsupported_arrow_column_type_is_a_data_error() -> None:
    frame = pl.DataFrame({"g": [datetime.date(2020, 1, 1)] * 2})
    with pytest.raises(gamfit.DataError, match="column 'g'"):
        normalize_table(frame)


def test_normalizing_a_table_imports_no_optional_table_library() -> None:
    script = textwrap.dedent(
        """
        import sys
        from gamfit._tables import normalize_table
        normalize_table({"x": [1.0, 2.0], "g": ["a", "b"]})
        loaded = sorted({"pandas", "polars", "pyarrow"} & set(sys.modules))
        assert not loaded, loaded
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
