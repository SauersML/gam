from __future__ import annotations

import importlib

pytest = importlib.import_module("pytest")
pytest.importorskip("gamfit._rust")


def test_compare_models_ranks_by_lowest_aic_with_deterministic_tie_break() -> None:
    import numpy as np
    import pandas as pd
    import gamfit

    rng = np.random.default_rng(123)
    x = np.linspace(-1.0, 1.0, 60)
    y = 1.0 + 1.5 * x + rng.normal(0.0, 0.06, size=x.shape[0])
    df = pd.DataFrame({"y": y, "x": x, "x2": x * x})

    m1 = gamfit.fit(df, "y ~ x")
    m2 = gamfit.fit(df, "y ~ x + x2")

    out = gamfit.compare_models([m1, m2], names=["linear", "quadratic"])
    rows = out["score_table"]

    row_by_name = {row["name"]: row for row in rows}
    expected = sorted(rows, key=lambda row: (float(row["aic"]), str(row["name"])))
    ranked_names = [entry[0] if isinstance(entry, (list, tuple)) else entry["name"] for entry in out["ranking"]]
    expected_names = [row["name"] for row in expected]

    assert ranked_names == expected_names, "compare_models ranking must order fits by lower AIC first and break ties deterministically"
    assert out["winner"] == expected_names[0], "winner must be the model with minimum AIC"
    assert float(row_by_name[expected_names[0]]["aic"]) <= float(row_by_name[expected_names[1]]["aic"]), "winner AIC must not exceed runner-up AIC"
