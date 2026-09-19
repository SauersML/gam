"""Regression test: a shape-constrained smooth's p-value is withheld with a reason.

A ``shape=`` smooth lives in the cone ``δ ≥ 0``, and the null ``f = 0`` is the
cone's apex. Under a flat truth the estimate sits on the boundary, so the Wald
χ² and the LR spectral reference do not describe the statistic's null law. The
boundary-aware references (chi-bar-square, active-set conditioning) are laws of
the cone projection. The engine's estimate is the truncated posterior mean,
which lies strictly inside the cone. So no calibrated p-value exists.

Before this change both surfaces dropped the number without saying why:
``summary().smooth_terms`` left ``p_value`` blank, and ``smooth_significance``
omitted the term entirely. A blank reads as "not significant" and a missing row
reads as "not a smooth". Every surface now names the typed reason, and the
unconstrained term beside it keeps its p-value.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


# A truth inside each shape's cone. The withheld p-value is a property of the
# term's shape, not of the data, so the case does not need a flat truth; a flat
# truth mostly ends in the outer-search refusal on the cone apex instead.
_TRUTH = {
    "monotone_increasing": lambda x: x,
    "monotone_decreasing": lambda x: -x,
    "convex": lambda x: (x - 0.5) ** 2,
    "concave": lambda x: -((x - 0.5) ** 2),
}


def _data(shape: str, seed: int, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x1) + _TRUTH[shape](x2) + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.mark.parametrize(
    "shape", ["monotone_increasing", "monotone_decreasing", "convex", "concave"]
)
def test_shape_constrained_pvalue_is_withheld_with_a_typed_reason(shape: str) -> None:
    df = _data(shape, seed=7)
    model = gamfit.fit(df, f"y ~ s(x1) + s(x2, shape={shape})")

    shaped_name = f"s(x2, shape={shape})"
    rows = {row["name"]: row for row in model.summary().smooth_terms}
    assert set(rows) == {"s(x1)", shaped_name}, rows
    free, shaped = rows["s(x1)"], rows[shaped_name]
    assert free.get("p_value") is not None, free
    assert "p_value_unavailable" not in free, free
    assert shaped.get("p_value") is None, shaped
    assert shaped.get("chi_sq") is None, shaped
    assert shaped["p_value_unavailable"] == "shape_constrained", shaped
    assert shaped["edf"] >= 0.0, shaped

    lr = model.smooth_significance(df)
    assert [row["term_idx"] for row in lr] == sorted(row["term_idx"] for row in lr)
    by_name = {row["name"]: row for row in lr}
    assert set(by_name) == {"s(x1)", shaped_name}, lr
    assert by_name["s(x1)"].get("p_value") is not None, by_name["s(x1)"]
    assert "p_value_unavailable" not in by_name["s(x1)"], by_name["s(x1)"]
    withheld = by_name[shaped_name]
    assert withheld["p_value_unavailable"] == "shape_constrained", withheld
    assert "shape-constrained" in withheld["explanation"], withheld
    assert set(withheld) == {"name", "term_idx", "p_value_unavailable", "explanation"}, withheld
