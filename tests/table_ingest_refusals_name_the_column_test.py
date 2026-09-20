"""Table-ingest refusals name the offending column and the numbers that fail.

A mapping whose columns disagree in length, a column that is not 1D, a column
that is not a vector at all, and a required column absent from the table are
each refused by ``gamfit._tables`` before the table reaches Rust. Each refusal
must say which column is wrong and by how much, and the data-shape refusals
must be the typed ``SchemaMismatchError`` the Rust encoder raises for the same
defect, so ``except gamfit.errors.DataError`` catches them.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit
from gamfit._tables import normalize_table


def test_unequal_column_lengths_name_both_columns_and_row_counts() -> None:
    with pytest.raises(gamfit.errors.SchemaMismatchError) as exc_info:
        normalize_table({"x": [1.0, 2.0], "y": [1.0, 2.0, 3.0]})
    assert str(exc_info.value) == "column 'y' has 3 rows but column 'x' has 2"


def test_two_dimensional_column_names_the_column_and_its_shape() -> None:
    with pytest.raises(gamfit.errors.SchemaMismatchError) as exc_info:
        normalize_table({"x": np.ones((3, 2)), "y": [1.0, 2.0, 3.0]})
    assert str(exc_info.value) == "column 'x' must be 1D, got 2 dimensions (shape (3, 2))"


def test_non_vector_column_names_the_column_and_its_type() -> None:
    with pytest.raises(TypeError) as exc_info:
        normalize_table({"x": {"a": 1.0}, "y": [1.0]})
    assert str(exc_info.value) == (
        "column 'x' must be a 1D array-like sequence of cells, got dict"
    )


def test_missing_required_column_names_it_and_the_available_columns() -> None:
    with pytest.raises(gamfit.errors.SchemaMismatchError) as exc_info:
        normalize_table({"x": [1.0, 2.0]}, required_columns=["x", "z"])
    assert str(exc_info.value) == (
        "required column 'z' not found in the input table; available columns: ['x']"
    )
