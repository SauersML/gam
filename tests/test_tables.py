from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol, cast


class _Pytest(Protocol):
    def raises(self, expected_exception: type[BaseException], *, match: str) -> Any: ...


pytest = cast(_Pytest, import_module("pytest"))

from gamfit._tables import (
    CATEGORICAL_CELL_SENTINEL,
    PredictionResult,
    normalize_table,
    restore_output_table,
)


def _rendered(table: Any) -> list[list[str]]:
    """Materialize an encoded table's rendered rows.

    `normalize_table` returns a native `_EncodedTable`, which is `frozen` and
    defines no `__eq__`, so comparing two of them compares object IDENTITY.
    Content comparisons must go through the rendered rows.
    """
    return [list(table[index]) for index in range(len(table))]


def test_normalize_table_dict_of_numpy_float64_arrays_renders_native_numbers() -> None:
    # Regression for #387: a numpy scalar must never reach the Rust core
    # carrying its NumPy 2.x repr ("np.float64(-3.0)"), which the core would
    # misread as a categorical level.
    #
    # The representation this test was originally written against is gone.
    # `normalize_table` no longer returns row-major Python strings: numeric
    # columns travel as ONE typed f64 block into `encoded_table_from_columns`
    # and are never stringified on the way in, so #387's hazard is now
    # structurally impossible for them. `_EncodedTable.__getitem__` renders a
    # row lazily, for display. What survives — and is asserted here — is the
    # property itself: no rendered cell carries a NumPy repr, and a numpy
    # column encodes identically to the equivalent Python-list column.
    import numpy as np

    x = np.array([-3.0, 0.5, 2.25], dtype=np.float64)
    y = np.array([1.0, -0.25, 3.5], dtype=np.float64)

    headers_np, rows_np, kind = normalize_table({"x": x, "y": y})

    assert kind == "mapping"
    assert headers_np == ["x", "y"]
    # No cell may carry the numpy type name; every cell must be a plain number.
    for row in rows_np:
        for cell in row:
            assert "np.float64" not in cell
            float(cell)  # must round-trip as a real number

    # Equivalent dict of Python-float lists must produce identical output.
    # This must compare CONTENT: `rows_np == rows_list` compared identity and
    # was therefore False for any two separately-built tables — an assertion
    # that could never hold, and so could never have caught a real divergence.
    headers_list, rows_list, _ = normalize_table(
        {"x": x.tolist(), "y": y.tolist()}
    )
    assert headers_np == headers_list
    assert _rendered(rows_np) == _rendered(rows_list)


def test_normalize_table_numpy_float16_and_int_scalars_render_natively() -> None:
    # float16 also subclasses float with a type-named repr in NumPy 2.x;
    # numpy integers must not arrive as "np.int64(3)".
    import numpy as np

    _, rows, _ = normalize_table(
        {
            "h": np.array([2.5, -1.0], dtype=np.float16),
            "i": np.array([3, -7], dtype=np.int64),
        }
    )

    flat = [cell for row in rows for cell in row]
    assert all("np." not in cell for cell in flat)
    for cell in flat:
        float(cell)

    # A numpy INTEGER column is a numeric covariate, not a factor: it carries no
    # categorical sentinel, and its VALUE survives exactly.
    #
    # It renders "3.0", not "3". Every non-categorical column is held as f64 and
    # rendered with Rust's `{:?}`, so an integral value shows a trailing `.0`.
    # The old `{rows[0][1], rows[1][1]} == {"3", "-7"}` pinned the spelling of a
    # row-major Python string table that no longer exists; against an f64 column
    # it is unsatisfiable no matter how exact the value is. The property worth
    # holding is exactness, and it is asserted numerically.
    assert all(not cell.startswith(CATEGORICAL_CELL_SENTINEL) for cell in flat)
    assert [float(rows[0][1]), float(rows[1][1])] == [3.0, -7.0]


def test_numpy_scalars_in_a_categorical_column_carry_no_numpy_repr() -> None:
    # #387's hazard has ONE surviving path, and this covers it. Numeric columns
    # never stringify, but a column that mixes a string with numerics ("object"
    # dtype in pandas) labels its numbers as levels. That is where "np.int64(1)"
    # could still reach the Rust core as a level name, so that is where the
    # guard belongs.
    import numpy as np

    headers, rows, _ = normalize_table(
        {
            "g": ["a", np.int64(1), np.float64(2.5)],
            "x": np.array([0.0, 1.0, 2.0], dtype=np.float64),
        }
    )

    g_index = headers.index("g")
    cells = [row[g_index] for row in rows]
    assert all(cell.startswith(CATEGORICAL_CELL_SENTINEL) for cell in cells), (
        f"a string+numeric column is categorical; cells were {cells!r}"
    )
    levels = [cell[len(CATEGORICAL_CELL_SENTINEL):] for cell in cells]
    assert levels == ["a", "1", "2.5"], (
        f"numpy scalars in a categorical column must render as bare values, "
        f"never a NumPy repr: {levels!r}"
    )


def test_normalize_table_rejects_zero_row_mapping() -> None:
    from gamfit.errors import DataError

    with pytest.raises(DataError, match="has no observations"):
        normalize_table({"x": [], "y": []})


def test_normalize_pandas_frame_excludes_row_index() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]},
        index=[11, 17, 29],
    )
    headers, rows, kind = normalize_table(frame)

    assert kind == "pandas"
    assert headers == ["x", "y"]
    assert rows.headers == headers
    assert rows.shape == (3, 2)
    assert list(rows) == [["1.0", "4.0"], ["2.0", "5.0"], ["3.0", "6.0"]]


def test_restore_output_table_rejects_unknown_return_type() -> None:
    with pytest.raises(ValueError, match="unsupported return_type 'arrowish'"):
        restore_output_table(
            {"mean": [1.0], "linear_predictor": [0.0]},
            requested="arrowish",
            input_kind="mapping",
            training_kind="unknown",
        )


def test_restore_output_table_dict_returns_prediction_result_with_field_access() -> None:
    restored = restore_output_table(
        {
            "linear_predictor": [0.5, 1.5],
            "mean": [1.0, 2.0],
            "std_error": [0.1, 0.2],
            "mean_lower": [0.8, 1.6],
            "mean_upper": [1.2, 2.4],
        },
        requested="dict",
        input_kind="mapping",
        training_kind="unknown",
    )

    assert isinstance(restored, dict)
    assert isinstance(restored, PredictionResult)
    assert list(restored) == [
        "linear_predictor",
        "mean",
        "std_error",
        "mean_lower",
        "mean_upper",
    ]
    assert restored["mean"] == [1.0, 2.0]
    assert restored.mean == [1.0, 2.0]
    assert restored.std_error == [0.1, 0.2]
    assert restored.mean_lower == [0.8, 1.6]
    assert restored.mean_upper == [1.2, 2.4]
    # Attribute access is by column name only: the historical `lower` /
    # `upper` / `se_mean` aliases are gone, because under two schemas
    # (`mean_lower` vs `posterior_mean_lower`) an alias names two columns.
    for alias in ("lower", "upper", "se_mean"):
        with pytest.raises(AttributeError, match=f"no prediction column '{alias}'"):
            _ = getattr(restored, alias)
    with pytest.raises(AttributeError, match="no prediction column 'median'"):
        _ = restored.median


def test_restore_output_table_supports_pyarrow_output() -> None:
    import pyarrow

    restored = restore_output_table(
        {"mean": [1.0, 2.0], "linear_predictor": [0.0, 0.5]},
        requested="pyarrow",
        input_kind="mapping",
        training_kind="unknown",
    )

    assert isinstance(restored, pyarrow.Table)
    assert restored.column_names == ["mean", "linear_predictor"]
    assert restored.to_pydict() == {"mean": [1.0, 2.0], "linear_predictor": [0.0, 0.5]}


def test_restore_output_table_prefers_pyarrow_training_kind() -> None:
    import pyarrow

    restored = restore_output_table(
        {"mean": [1.0], "linear_predictor": [0.0]},
        requested=None,
        input_kind="mapping",
        training_kind="pyarrow",
    )

    assert isinstance(restored, pyarrow.Table)


# --- #1467 / #1468 / #1469: dict / records / numpy numeric-string labels ---
# A dict (or records / numpy) column whose values are Python `str` must be
# detected as categorical, matching pandas string/object-dtype behavior — so
# numeric-string labels ("0", "1", "2") are NOT inferred numeric. A column of
# int/float stays numeric (a genuinely-numeric by= covariate is preserved).
def _categorical_columns(data: Any) -> set[str]:
    """Columns the encoded table holds as categorical (sentinel-rendered)."""
    headers, rows, _ = normalize_table(data)
    first = rows[0]
    return {
        header
        for header, cell in zip(headers, first)
        if cell.startswith(CATEGORICAL_CELL_SENTINEL)
    }


def test_dict_numeric_string_column_is_categorical() -> None:
    data = {"g": ["0", "1", "2", "0", "1"], "y": [1.0, 2.0, 3.0, 4.0, 5.0]}
    assert _categorical_columns(data) == {"g"}


def test_dict_numeric_covariate_stays_numeric() -> None:
    # A genuinely-numeric by= covariate supplied as floats must NOT be treated
    # as categorical — only string-valued columns are.
    assert _categorical_columns({"age": [25.0, 30.0, 35.5], "y": [1.0, 2.0, 3.0]}) == set()


def test_records_numeric_string_column_is_categorical() -> None:
    recs = [{"g": "0", "y": 1.0}, {"g": "1", "y": 2.0}, {"g": "2", "y": 3.0}]
    assert _categorical_columns(recs) == {"g"}


# A MIXED string+numeric column is `object` dtype in pandas (→ categorical). The
# untyped (dict/records/numpy) value-inference must agree, or it is the same
# typed-vs-untyped parity gap as #1467/#1468/#1469 one boundary further out: the
# old "every non-null value must be str" rule saw the numeric and lowered the
# whole column to a NUMERIC covariate, dropping the string levels.
def test_dict_mixed_string_numeric_column_is_categorical() -> None:
    assert _categorical_columns({"g": ["a", 1, "b", 2], "y": [1.0, 2.0, 3.0, 4.0]}) == {"g"}


def test_dict_numeric_column_with_one_string_is_categorical() -> None:
    # The dual: a numeric column with a single stray string ("NA") is object in
    # pandas (→ categorical); it must NOT be silently treated as numeric.
    assert _categorical_columns({"v": [1.0, 2.0, "NA", 4.0]}) == {"v"}


def test_pure_bool_column_stays_numeric_but_bool_str_is_categorical() -> None:
    # `bool` is not a `str`: a pure-bool column is `bool` dtype in pandas (numeric);
    # a bool+str mix is `object` (categorical). Pins both halves so the string
    # rule does not accidentally sweep in pure-bool columns.
    data = {"flag": [True, False, True], "mixed": [True, "yes", False]}
    assert _categorical_columns(data) == {"mixed"}


def test_pandas_numeric_string_column_is_categorical() -> None:
    import pandas as pd

    df = pd.DataFrame({"g": ["0", "1", "2"], "y": [1.0, 2.0, 3.0]})
    assert _categorical_columns(df) == {"g"}


def test_normalize_table_dict_numeric_string_stamps_categorical_sentinel() -> None:
    # End-to-end: a dict numeric-string-label column must reach normalize_table's
    # row output with the categorical sentinel prefix (so Rust force_categorical
    # fires), while a float column stays plain numeric text.
    from gamfit._tables import CATEGORICAL_CELL_SENTINEL, normalize_table

    headers, rows, kind = normalize_table(
        {"g": ["0", "1", "2"], "y": [1.0, 2.0, 3.0]}
    )
    assert kind == "mapping"
    g_index = headers.index("g")
    y_index = headers.index("y")
    for row in rows:
        assert row[g_index].startswith(CATEGORICAL_CELL_SENTINEL), (
            f"dict numeric-string cell must carry the categorical sentinel: {row[g_index]!r}"
        )
        assert not row[y_index].startswith(CATEGORICAL_CELL_SENTINEL), (
            f"float column must stay plain numeric: {row[y_index]!r}"
        )
