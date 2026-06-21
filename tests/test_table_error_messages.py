"""Test improved error messages in table normalization.

This test verifies that NaN/None/empty cell errors include column and row context,
making them actionable for users debugging their data.
"""

from __future__ import annotations

import pytest


def test_nan_error_includes_column_and_row() -> None:
    """NaN values should report which column and row they occur in."""
    # Skip if pandas not available
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")

    from gamfit._tables import normalize_table

    # Create a DataFrame with NaN in the 'x' column, row 2 (0-indexed)
    df = pd.DataFrame({
        "x": [1.0, 2.0, float("nan"), 4.0],
        "y": [10.0, 20.0, 30.0, 40.0],
    })

    with pytest.raises(ValueError) as exc_info:
        normalize_table(df)

    error_msg = str(exc_info.value)
    assert "x" in error_msg, f"Expected column name 'x' in error: {error_msg}"
    assert "row 3" in error_msg, f"Expected row 3 in error: {error_msg}"


def test_none_error_includes_context() -> None:
    """None values should report which column and row they occur in."""
    pytest.importorskip("pandas")

    from gamfit._tables import normalize_table

    # Using a dict input that can have None
    data = {
        "x": [1.0, 2.0, None, 4.0],
        "y": [10.0, 20.0, 30.0, 40.0],
    }

    with pytest.raises(ValueError) as exc_info:
        normalize_table(data)

    error_msg = str(exc_info.value)
    assert "x" in error_msg, f"Expected column name 'x' in error: {error_msg}"
    assert "row 3" in error_msg, f"Expected row 3 in error: {error_msg}"


def test_stringify_cell_direct_call_still_works() -> None:
    """Direct calls to stringify_cell should work with or without context."""
    from gamfit._tables import stringify_cell

    # Without context (backward compatible)
    assert stringify_cell(3.14) == "3.14"
    assert stringify_cell(42) == "42"
    assert stringify_cell(True) == "1"

    # With context (optional improvement)
    assert stringify_cell(3.14, column="test_col") == "3.14"
    assert stringify_cell(42, column="age", row=5) == "42"


def test_dask_data_frame_support() -> None:
    """Dask DataFrames should be accepted as input."""
    pd = pytest.importorskip("pandas")
    dd = pytest.importorskip("dask.dataframe")

    # Create a Dask DataFrame from pandas
    pdf = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    ddf = dd.from_pandas(pdf, npartitions=2)

    from gamfit._tables import normalize_table, detect_table_kind

    # Test detection
    assert detect_table_kind(ddf) == "dask"

    # Test normalization (should work)
    headers, rows, kind = normalize_table(ddf)
    assert kind == "dask"
    assert headers == ["x", "y"]
    assert len(rows) == 5


def test_dask_error_message_includes_context() -> None:
    """Dask DataFrames with NaN should report column and row context."""
    pd = pytest.importorskip("pandas")
    dd = pytest.importorskip("dask.dataframe")

    from gamfit._tables import normalize_table

    # Create a Dask DataFrame with NaN
    pdf = pd.DataFrame({
        "x": [1.0, 2.0, float("nan"), 4.0],
        "y": [10.0, 20.0, 30.0, 40.0],
    })
    ddf = dd.from_pandas(pdf, npartitions=2)

    with pytest.raises(ValueError) as exc_info:
        normalize_table(ddf)

    error_msg = str(exc_info.value)
    assert "x" in error_msg, f"Expected column name 'x' in error: {error_msg}"
    assert "row 3" in error_msg, f"Expected row 3 in error: {error_msg}"
