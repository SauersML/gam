from __future__ import annotations

import importlib

pytest = importlib.import_module("pytest")
pytest.importorskip("gamfit._rust")


def test_failure_at_returns_monotone_decreasing_survival_in_time() -> None:
    import numpy as np
    import pandas as pd
    import gamfit

    rng = np.random.default_rng(202)
    n = 120
    x = rng.normal(0.0, 1.0, size=n)
    entry = np.zeros(n)
    event_time = rng.exponential(scale=2.0, size=n)
    censor_time = rng.exponential(scale=3.0, size=n)
    exit_age = np.maximum(np.minimum(event_time, censor_time), 1e-3)
    event = (event_time <= censor_time).astype(int)
    df = pd.DataFrame({"entry": entry, "exit": exit_age, "event": event, "x": x})

    model = gamfit.fit(df, "Surv(entry, exit, event) ~ x")
    pred = model.predict(df.iloc[:3].copy())

    assert hasattr(pred, "failure_at"), "SurvivalPrediction must expose failure_at(t)"
    t = np.linspace(0.05, float(np.max(exit_age)), 40)
    survival = 1.0 - np.asarray(pred.failure_at(t), dtype=float)
    diffs = np.diff(survival, axis=1)
    assert np.all(diffs <= 1e-10), "Survival probability returned through failure_at must be monotonically non-increasing in t"
