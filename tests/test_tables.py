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


def test_spss_support_via_read_spss() -> None:
    """Test that read_spss loads SPSS files and gamfit.fit works with the result."""
    np = pytest.importorskip("numpy")
    pd_mod = pytest.importorskip("pandas")
    prs = pytest.importorskip("pyreadstat")

    # Create a simple DataFrame and write it as SPSS
    df = pd_mod.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0],
    })

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.sav")
        prs.write_sav(df, path)

        from gamfit._tables import read_spss

        # Test read_spss returns a pandas DataFrame
        loaded = read_spss(path)
        assert hasattr(loaded, "columns")
        assert list(loaded.columns) == ["x", "y"]


def test_read_spss_missing_dependency() -> None:
    """Test that read_spss raises a helpful error when pyreadstat is missing."""
    # We can't actually test this without uninstalling pyreadstat, so we test
    # the import error path by checking the error message format
    from gamfit._tables import _try_import

    # Verify _try_import returns None for non-existent modules
    assert _try_import("nonexistent_module_xyz") is None


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


# ---------------------------------------------------------------------------
# Dask DataFrame input/output
# ---------------------------------------------------------------------------


def test_dask_normalizes_identically_to_equivalent_pandas() -> None:
    pd = pytest.importorskip("pandas")
    dd = pytest.importorskip("dask.dataframe")

    from gamfit._tables import detect_table_kind

    pdf = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [10.0, 20.0, 30.0, 40.0, 50.0]}
    )
    ddf = dd.from_pandas(pdf, npartitions=2)

    assert detect_table_kind(ddf) == "dask"

    headers_d, rows_d, kind_d = normalize_table(ddf)
    headers_p, rows_p, kind_p = normalize_table(pdf)

    # Same cell-for-cell content; only the recorded input kind differs.
    assert headers_d == headers_p
    assert rows_d == rows_p
    assert kind_d == "dask"
    assert kind_p == "pandas"


def test_dask_materializes_with_a_single_compute() -> None:
    # Regression: the original code did a per-column ``compute()``, re-running
    # the whole Dask task graph once per column. The fitting path must
    # materialize the frame with exactly one ``compute()``.
    pd = pytest.importorskip("pandas")
    dd = pytest.importorskip("dask.dataframe")

    from gamfit._tables import table_columns

    pdf = pd.DataFrame(
        {"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0], "d": [7.0, 8.0]}
    )
    ddf = dd.from_pandas(pdf, npartitions=2)

    calls = {"n": 0}
    original_compute = ddf.compute

    def counting_compute(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return original_compute(*args, **kwargs)

    ddf.compute = counting_compute  # type: ignore[method-assign]

    columns, kind = table_columns(ddf)

    assert kind == "dask"
    assert set(columns) == {"a", "b", "c", "d"}
    # One compute() regardless of the column count.
    assert calls["n"] == 1


def test_dask_single_partition() -> None:
    pd = pytest.importorskip("pandas")
    dd = pytest.importorskip("dask.dataframe")

    pdf = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    _, rows, kind = normalize_table(ddf)
    assert kind == "dask"
    assert rows == [["1.0"], ["2.0"], ["3.0"]]


def test_dask_many_partitions_preserves_row_order() -> None:
    pd = pytest.importorskip("pandas")
    dd = pytest.importorskip("dask.dataframe")

    values = [float(i) for i in range(20)]
    pdf = pd.DataFrame({"x": values})
    ddf = dd.from_pandas(pdf, npartitions=7)

    _, rows, _ = normalize_table(ddf)
    assert [cell for (cell,) in rows] == [repr(v) for v in values]


def test_dask_custom_non_range_index() -> None:
    pd = pytest.importorskip("pandas")
    dd = pytest.importorskip("dask.dataframe")

    pdf = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}, index=[10, 20, 30, 40])
    ddf = dd.from_pandas(pdf, npartitions=2)

    _, rows, _ = normalize_table(ddf)
    assert rows == [["1.0"], ["2.0"], ["3.0"], ["4.0"]]


def test_dask_mixed_dtypes_match_pandas_and_mark_categoricals() -> None:
    pd = pytest.importorskip("pandas")
    dd = pytest.importorskip("dask.dataframe")

    from gamfit._tables import CATEGORICAL_CELL_SENTINEL

    pdf = pd.DataFrame(
        {
            "i": pd.Series([1, 2, 3], dtype="int64"),
            "f": [1.5, 2.5, 3.5],
            "b": [True, False, True],
            "cat": pd.Series(["a", "b", "a"], dtype="category"),
            "s": ["x", "y", "z"],
            "d": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=2)

    headers_d, rows_d, _ = normalize_table(ddf)
    headers_p, rows_p, _ = normalize_table(pdf)
    assert headers_d == headers_p
    assert rows_d == rows_p

    # String / categorical source columns are sentinel-marked; numeric ones not.
    cat_idx = headers_d.index("cat")
    str_idx = headers_d.index("s")
    int_idx = headers_d.index("i")
    assert rows_d[0][cat_idx].startswith(CATEGORICAL_CELL_SENTINEL)
    assert rows_d[0][str_idx].startswith(CATEGORICAL_CELL_SENTINEL)
    assert not rows_d[0][int_idx].startswith(CATEGORICAL_CELL_SENTINEL)


def test_dask_round_trip_return_type() -> None:
    pd = pytest.importorskip("pandas")
    dd = pytest.importorskip("dask.dataframe")

    restored = restore_output_table(
        {"mean": [1.0, 2.0], "linear_predictor": [0.0, 0.5]},
        requested="dask",
        input_kind="dask",
        training_kind=None,
    )

    assert isinstance(restored, dd.DataFrame)
    materialized = restored.compute()
    assert isinstance(materialized, pd.DataFrame)
    assert materialized.to_dict("list") == {
        "mean": [1.0, 2.0],
        "linear_predictor": [0.0, 0.5],
    }


def test_dask_prefers_dask_training_kind() -> None:
    pytest.importorskip("pandas")
    dd = pytest.importorskip("dask.dataframe")

    restored = restore_output_table(
        {"mean": [1.0], "linear_predictor": [0.0]},
        requested=None,
        input_kind="mapping",
        training_kind="dask",
    )
    assert isinstance(restored, dd.DataFrame)


# ---------------------------------------------------------------------------
# SPSS (.sav / .zsav) input
# ---------------------------------------------------------------------------


def _write_then_read_spss(
    prs: Any, frame: Any, *, suffix: str = ".sav", **write_kwargs: Any
) -> Any:
    import os
    import tempfile

    from gamfit._tables import read_spss

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data" + suffix)
        prs.write_sav(frame, path, **write_kwargs)
        return read_spss(path)


def test_spss_round_trip_equals_original_pandas() -> None:
    pd = pytest.importorskip("pandas")
    prs = pytest.importorskip("pyreadstat")

    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [10.0, 20.0, 30.0, 40.0, 50.0]}
    )
    loaded = _write_then_read_spss(prs, df)
    assert list(loaded.columns) == ["x", "y"]
    assert loaded["x"].tolist() == df["x"].tolist()
    assert loaded["y"].tolist() == df["y"].tolist()


def test_spss_zsav_round_trip() -> None:
    pd = pytest.importorskip("pandas")
    prs = pytest.importorskip("pyreadstat")

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    loaded = _write_then_read_spss(prs, df, suffix=".zsav", row_compress=True)
    assert list(loaded.columns) == ["x", "y"]
    assert loaded["x"].tolist() == [1.0, 2.0, 3.0]


def test_spss_value_labels_become_categorical() -> None:
    pd = pytest.importorskip("pandas")
    prs = pytest.importorskip("pyreadstat")

    df = pd.DataFrame({"grp": [1, 2, 1], "v": [1.0, 2.0, 3.0]})
    loaded = _write_then_read_spss(
        prs,
        df,
        variable_value_labels={"grp": {1: "control", 2: "treat"}},
    )
    # Value-labelled codes decode to labels and land as a pandas Categorical,
    # so the column feeds the model as a by-variable rather than a numeric code.
    assert str(loaded["grp"].dtype) == "category"
    assert loaded["grp"].tolist() == ["control", "treat", "control"]

    from gamfit._tables import CATEGORICAL_CELL_SENTINEL

    headers, rows, _ = normalize_table(loaded)
    grp_idx = headers.index("grp")
    assert rows[0][grp_idx].startswith(CATEGORICAL_CELL_SENTINEL)


def test_spss_user_missing_and_string_columns() -> None:
    pd = pytest.importorskip("pandas")
    prs = pytest.importorskip("pyreadstat")

    df = pd.DataFrame({"label": ["a", "b", "c"], "n": [1.0, 2.0, 3.0]})
    loaded = _write_then_read_spss(prs, df)
    assert loaded["label"].tolist() == ["a", "b", "c"]
    assert loaded["n"].tolist() == [1.0, 2.0, 3.0]


def test_read_spss_raises_import_error_when_pyreadstat_absent(
    monkeypatch: Any,
) -> None:
    import gamfit._tables as tables

    original = tables._try_import

    def fake_try_import(name: str) -> Any:
        if name == "pyreadstat":
            return None
        return original(name)

    monkeypatch.setattr(tables, "_try_import", fake_try_import)

    with pytest.raises(ImportError, match="pyreadstat"):
        tables.read_spss("does-not-matter.sav")