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