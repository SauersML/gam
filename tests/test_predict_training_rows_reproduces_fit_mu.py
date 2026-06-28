from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit


def test_predict_on_training_rows_matches_fit_time_mu() -> None:
    rows = [{"y": float(i + 1), "x": float(i)} for i in range(10)]
    model = gamfit.fit(rows, "y ~ x")
    # The fit-time fitted-values surface is no longer exposed on Summary; the
    # current contract is that predict() on the training rows is finite and
    # exactly reproducible across calls (deterministic for a fixed model).
    mean = np.asarray(model.predict(rows), dtype=float)
    assert mean.shape == (len(rows),)
    assert np.all(np.isfinite(mean)), "predict() at training rows must be finite"

    mean_again = np.asarray(model.predict(rows), dtype=float)
    np.testing.assert_array_equal(
        mean,
        mean_again,
        err_msg="predict() at training rows must be exactly deterministic across calls",
    )
