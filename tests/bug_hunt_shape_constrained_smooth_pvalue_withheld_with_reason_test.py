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


def _data(seed: int, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    # x2 is flat: the shape term's null is true, the case the reference fails on.
    y = np.sin(2.0 * np.pi * x1) + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.mark.parametrize(
    "shape", ["monotone_increasing", "monotone_decreasing", "convex", "concave"]
)
def test_shape_constrained_pvalue_is_withheld_with_a_typed_reason(shape: str) -> None:
    df = _data(seed=11)
    model = gamfit.fit(df, f"y ~ s(x1) + s(x2, shape={shape})")

    rows = {row["name"]: row for row in model.summary().smooth_terms}
    assert set(rows) == {"s(x1)", "s(x2)"}, rows
    free, shaped = rows["s(x1)"], rows["s(x2)"]
    assert free.get("p_value") is not None, free
    assert "p_value_unavailable" not in free, free
    assert shaped.get("p_value") is None, shaped
    assert shaped.get("chi_sq") is None, shaped
    assert shaped["p_value_unavailable"] == "shape_constrained", shaped
    assert shaped["edf"] >= 0.0, shaped

    lr = model.smooth_significance(df)
    assert [row["term_idx"] for row in lr] == sorted(row["term_idx"] for row in lr)
    by_name = {row["name"]: row for row in lr}
    assert set(by_name) == {"s(x1)", "s(x2)"}, lr
    assert by_name["s(x1)"].get("p_value_corrected") is not None, by_name["s(x1)"]
    assert "p_value_unavailable" not in by_name["s(x1)"], by_name["s(x1)"]
    withheld = by_name["s(x2)"]
    assert withheld["p_value_unavailable"] == "shape_constrained", withheld
    assert "shape-constrained" in withheld["explanation"], withheld
    assert set(withheld) == {"name", "term_idx", "p_value_unavailable", "explanation"}, withheld
