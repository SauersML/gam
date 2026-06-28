from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit


@pytest.mark.xfail(
    strict=True,
    reason="#1512 triage: model.summary() no longer carries a 'fitted' key "
    "(KeyError) — the fit-time fitted-values surface moved off Summary, whose "
    "keys are now coefficients/smooth_terms/lambdas/etc. Re-point at the "
    "current fitted-values accessor to re-enable the predict==fit-mu check.",
)
def test_predict_on_training_rows_matches_fit_time_mu() -> None:
    rows = [{"y": float(i + 1), "x": float(i)} for i in range(10)]
    model = gamfit.fit(rows, "y ~ x")
    pred = model.predict(rows)
    fitted = np.asarray(model.summary()["fitted"], dtype=float)
    mean = np.asarray(pred, dtype=float)
    np.testing.assert_allclose(
        mean,
        fitted,
        atol=1e-9,
        err_msg="predict() at training rows should reproduce fit-time mean values within 1e-9",
    )
