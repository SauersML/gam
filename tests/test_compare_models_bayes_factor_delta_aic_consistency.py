from __future__ import annotations

import importlib
import math

pytest = importlib.import_module("pytest")
pytest.importorskip("gamfit._rust")


def test_compare_models_bayes_factor_and_delta_aic_match_log_likelihood() -> None:
    import numpy as np
    import pandas as pd
    import gamfit

    rng = np.random.default_rng(9)
    x = np.linspace(-1.5, 1.5, 72)
    y = 0.3 + 1.2 * x + rng.normal(0.0, 0.08, size=x.shape[0])
    df = pd.DataFrame({"y": y, "x": x, "x2": x * x})

    m1 = gamfit.fit(df, "y ~ x")
    m2 = gamfit.fit(df, "y ~ x + x2")

    out = gamfit.compare_models([m1, m2], names=["m1", "m2"])
    rows = {row["name"]: row for row in out["score_table"]}

    ll1 = float(rows["m1"]["log_likelihood"])
    ll2 = float(rows["m2"]["log_likelihood"])
    k1 = float(rows["m1"].get("edf", rows["m1"].get("k", 0.0)))
    k2 = float(rows["m2"].get("edf", rows["m2"].get("k", 0.0)))

    expected_delta_aic = (2.0 * k2 - 2.0 * ll2) - (2.0 * k1 - 2.0 * ll1)
    actual_delta_aic = float(rows["m2"]["delta_aic"]) - float(rows["m1"]["delta_aic"])
    assert abs(actual_delta_aic - expected_delta_aic) < 1e-6, "delta_aic must be numerically consistent with log-likelihood and complexity penalty"

    expected_bf = math.exp(ll1 - ll2)
    actual_bf = float(rows["m1"]["bayes_factor"]) / max(float(rows["m2"]["bayes_factor"]), 1e-300)
    assert abs(actual_bf - expected_bf) / max(expected_bf, 1e-12) < 1e-6, "bayes_factor ratio must equal exp(loglik difference)"
