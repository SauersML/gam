from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gamfit._rust")

import gamfit


def _frame(n: int = 40) -> pd.DataFrame:
    x = np.linspace(0.0, 1.0, n)
    g = np.where(np.arange(n) % 2 == 0, "A", "B")
    y = np.sin(2 * np.pi * x) + (g == "B") * (0.25 + 0.5 * x)
    return pd.DataFrame({"y": y, "x": x, "g": g})


def test_unordered_by_factor_smooth_and_difference_api_smoke() -> None:
    df = _frame()
    model = gamfit.fit(df, "y ~ s(x, k=6, by=g)")
    out = model.difference_smooth(view="x", group="g", n=12, n_sim=128, simultaneous=True)
    assert len(out) == 12
    assert set(["x", "level_1", "level_2", "diff", "se", "lower", "upper", "critical"]).issubset(out.columns)
    assert np.all(np.isfinite(out["diff"]))
    assert np.all(out["se"] >= 0)
    assert float(out["critical"].iloc[0]) >= 1.0

