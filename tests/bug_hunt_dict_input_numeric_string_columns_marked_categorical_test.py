"""Bug hunt (pure-Python, no compiled extension): a dict/records/numpy column
whose values are all strings must be stamped with the categorical sentinel by
``normalize_table`` — even when the labels look numeric ("0"/"1"/"2").

This is the root-cause unit regression for #1467 (by= factor smooth lowered to
numeric), #1468 (categorical main effect collapsed to a numeric line) and #1469
(multinomial response wrongly rejected as numeric). All three share one cause:
``categorical_dtype_columns`` only marked typed (pandas/polars/pyarrow) columns,
so an all-string column from a dict reached the Rust core UNMARKED and was
inferred numeric. The fix infers categoricality from VALUE TYPE for untyped
inputs: a column is categorical iff non-empty and every non-null value is a
``str``.

This test does NOT import the compiled extension, so it is a cheap, fast guard
that fails before the fix (the ``g`` cells are unmarked) and passes after.
"""

from __future__ import annotations

from gamfit._tables import CATEGORICAL_CELL_SENTINEL, normalize_table


def test_dict_numeric_string_column_marked_numeric_column_not() -> None:
    headers, rows, _kind = normalize_table({"g": ["0", "1", "2"], "y": [1.0, 2.0, 3.0]})
    g_idx = headers.index("g")
    y_idx = headers.index("y")

    # Every cell of the all-string `g` column (numeric-looking labels) must be
    # marked categorical, so the Rust core builds a factor rather than a numeric
    # covariate.
    for row in rows:
        assert row[g_idx].startswith(CATEGORICAL_CELL_SENTINEL), (
            f"string column 'g' cell {row[g_idx]!r} was not marked categorical; "
            f"numeric-looking labels would be lowered to a numeric covariate"
        )

    # The genuinely-numeric `y` column must NOT be marked.
    for row in rows:
        assert not row[y_idx].startswith(CATEGORICAL_CELL_SENTINEL), (
            f"numeric column 'y' cell {row[y_idx]!r} was wrongly marked categorical"
        )


def test_dict_alphabetic_string_column_also_marked() -> None:
    headers, rows, _kind = normalize_table({"g": ["a", "b", "a"], "y": [1.0, 2.0, 3.0]})
    g_idx = headers.index("g")
    for row in rows:
        assert row[g_idx].startswith(CATEGORICAL_CELL_SENTINEL), (
            f"alphabetic string column cell {row[g_idx]!r} was not marked categorical"
        )
