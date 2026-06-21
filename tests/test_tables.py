from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol, cast


class _Pytest(Protocol):
    def importorskip(self, modname: str) -> Any: ...

    def raises(self, expected_exception: type[BaseException], *, match: str) -> Any: ...


pytest = cast(_Pytest, import_module("pytest"))

from gamfit._tables import PredictionResult, normalize_table, restore_output_table


def test_normalize_table_dict_of_numpy_float64_arrays_renders_native_numbers() -> None:
    # Regression for #387: a dict of numpy float64 arrays must stringify to
    # bare numeric text ("-3.0"), not the NumPy 2.x scalar repr
    # ("np.float64(-3.0)") which the Rust core misreads as a categorical level.
    np = pytest.importorskip("numpy")

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
    headers_list, rows_list, _ = normalize_table(
        {"x": x.tolist(), "y": y.tolist()}
    )
    assert headers_np == headers_list
    assert rows_np == rows_list


def test_normalize_table_numpy_float16_and_int_scalars_render_natively() -> None:
    # float16 also subclasses float with a type-named repr in NumPy 2.x;
    # numpy integers must render as bare integers, not "np.int64(3)".
    np = pytest.importorskip("numpy")

    _, rows, _ = normalize_table(
        {
            "h": np.array([2.5, -1.0], dtype=np.float16),
            "i": np.array([3, -7], dtype=np.int64),
        }
    )

    flat = [cell for row in rows for cell in row]
    assert all("np." not in cell for cell in flat)
    assert {rows[0][1], rows[1][1]} == {"3", "-7"}
    for cell in flat:
        float(cell)


def test_normalize_table_rejects_zero_row_mapping() -> None:
    with pytest.raises(ValueError, match="table data cannot be empty"):
        normalize_table({"x": [], "y": []})


def test_restore_output_table_rejects_unknown_return_type() -> None:
    with pytest.raises(ValueError, match="unsupported return_type 'arrowish'"):
        restore_output_table(
            {"mean": [1.0], "linear_predictor": [0.0]},
            requested="arrowish",
            input_kind="mapping",
            training_kind=None,
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
        training_kind=None,
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
    assert restored.lower == [0.8, 1.6]
    assert restored.upper == [1.2, 2.4]
    assert restored.se_mean == [0.1, 0.2]
    with pytest.raises(AttributeError, match="no prediction column 'median'"):
        _ = restored.median


def test_restore_output_table_supports_pyarrow_output() -> None:
    pyarrow = pytest.importorskip("pyarrow")

    restored = restore_output_table(
        {"mean": [1.0, 2.0], "linear_predictor": [0.0, 0.5]},
        requested="pyarrow",
        input_kind="mapping",
        training_kind=None,
    )

    assert isinstance(restored, pyarrow.Table)
    assert restored.column_names == ["mean", "linear_predictor"]
    assert restored.to_pydict() == {"mean": [1.0, 2.0], "linear_predictor": [0.0, 0.5]}


def test_restore_output_table_prefers_pyarrow_training_kind() -> None:
    pyarrow = pytest.importorskip("pyarrow")

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
def test_categorical_dtype_columns_dict_numeric_string_is_categorical() -> None:
    from gamfit._tables import categorical_dtype_columns, table_columns

    data = {"g": ["0", "1", "2", "0", "1"], "y": [1.0, 2.0, 3.0, 4.0, 5.0]}
    columns, kind = table_columns(data)
    categorical = categorical_dtype_columns(data, kind, columns=columns)
    assert "g" in categorical, f"numeric-string column must be categorical: {categorical}"
    assert "y" not in categorical, f"float column must stay numeric: {categorical}"


def test_categorical_dtype_columns_dict_numeric_covariate_stays_numeric() -> None:
    from gamfit._tables import categorical_dtype_columns, table_columns

    # A genuinely-numeric by= covariate supplied as floats must NOT be treated
    # as categorical — only string-valued columns are.
    data = {"age": [25.0, 30.0, 35.5], "y": [1.0, 2.0, 3.0]}
    columns, kind = table_columns(data)
    categorical = categorical_dtype_columns(data, kind, columns=columns)
    assert categorical == frozenset(), f"numeric columns must stay numeric: {categorical}"


def test_categorical_dtype_columns_records_numeric_string_is_categorical() -> None:
    from gamfit._tables import categorical_dtype_columns, table_columns

    recs = [{"g": "0", "y": 1.0}, {"g": "1", "y": 2.0}, {"g": "2", "y": 3.0}]
    columns, kind = table_columns(recs)
    categorical = categorical_dtype_columns(recs, kind, columns=columns)
    assert "g" in categorical, f"records numeric-string must be categorical: {categorical}"


def test_categorical_dtype_columns_pandas_numeric_string_still_categorical() -> None:
    pd = pytest.importorskip("pandas")
    from gamfit._tables import categorical_dtype_columns, table_columns

    df = pd.DataFrame({"g": ["0", "1", "2"], "y": [1.0, 2.0, 3.0]})
    columns, kind = table_columns(df)
    categorical = categorical_dtype_columns(df, kind, columns=columns)
    assert "g" in categorical, f"pandas regression guard: {categorical}"


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
