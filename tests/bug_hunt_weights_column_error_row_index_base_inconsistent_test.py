"""Bug hunt #1597: ``weights=`` column validators disagreed on the row-index
base, so the SAME physical row was named by two different row numbers
depending on which validation a bad value tripped.

``gamfit.fit(..., weights="w")`` runs two separate weight-column checks:

* the **non-finite** (NaN/inf) check in the Python ingestion layer
  (``gamfit/_tables.py``), which reports a **1-based** row, and
* the **non-negative** check in the Rust materializer
  (``crates/gam-models/.../materialize/columns.rs``), which reported a
  **0-based** row.

A row index in a user-facing error must identify the row unambiguously, so a
NaN and a negative weight at the *same* array index must produce the same row
number. The project standardizes on 1-based row reporting (every ``gam-data``
ingestion validator and the Python ``_tables.py`` siblings already use
``row + 1``); the fix aligns the Rust weight checks to that convention.

This test places a negative weight and (separately) a NaN weight at the SAME
array index, parses the ``row N`` token from each error, and asserts the two
row numbers agree and equal the 1-based index. It fails on the pre-fix tree
(``'row 2'`` vs ``'row 3'``) and passes once both validators use the 1-based
convention.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

import gamfit


def _row_number(message: str) -> int:
    """Extract the integer following the ``row`` token in an error message."""
    match = re.search(r"\brow\s+(\d+)\b", message)
    assert match is not None, f"no `row N` token in error message: {message}"
    return int(match.group(1))


def _fit_error(weights: list[float]) -> str:
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "w": weights,
        }
    )
    try:
        gamfit.fit(df, "y ~ x", weights="w")
    except Exception as exc:  # noqa: BLE001 - we assert on the message text
        return str(exc)
    raise AssertionError("expected fit to reject the bad weights column")


def test_negative_and_nan_weight_report_same_one_based_row() -> None:
    # Both bad values sit at array index 2, i.e. the 3rd row (1-based).
    neg_msg = _fit_error([1.0, 1.0, -1.0, 1.0, 1.0])
    nan_msg = _fit_error([1.0, 1.0, np.nan, 1.0, 1.0])

    neg_row = _row_number(neg_msg)
    nan_row = _row_number(nan_msg)

    # Convention chosen: 1-based (the majority across gam-data + _tables.py).
    assert neg_row == 3, f"negative-weight error must report 1-based row 3: {neg_msg!r}"
    assert nan_row == 3, f"non-finite-weight error must report 1-based row 3: {nan_msg!r}"
    assert neg_row == nan_row, (
        "negative and non-finite weight checks must agree on the row number: "
        f"{neg_msg!r} vs {nan_msg!r}"
    )
