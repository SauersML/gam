from __future__ import annotations

import pytest

from gam._tables import normalize_table, restore_output_table


def test_normalize_table_rejects_zero_row_mapping():
    with pytest.raises(ValueError, match="table data cannot be empty"):
        normalize_table({"x": [], "y": []})


def test_restore_output_table_rejects_unknown_return_type():
    with pytest.raises(ValueError, match="unsupported return_type 'arrowish'"):
        restore_output_table(
            {"mean": [1.0], "eta": [0.0]},
            requested="arrowish",
            input_kind="mapping",
            training_kind=None,
        )


def test_restore_output_table_supports_pyarrow_output():
    pyarrow = pytest.importorskip("pyarrow")

    restored = restore_output_table(
        {"mean": [1.0, 2.0], "eta": [0.0, 0.5]},
        requested="pyarrow",
        input_kind="mapping",
        training_kind=None,
    )

    assert isinstance(restored, pyarrow.Table)
    assert restored.column_names == ["mean", "eta"]
    assert restored.to_pydict() == {"mean": [1.0, 2.0], "eta": [0.0, 0.5]}


def test_restore_output_table_prefers_pyarrow_training_kind():
    pyarrow = pytest.importorskip("pyarrow")

    restored = restore_output_table(
        {"mean": [1.0], "eta": [0.0]},
        requested=None,
        input_kind="mapping",
        training_kind="pyarrow",
    )

    assert isinstance(restored, pyarrow.Table)
