"""Regression for issue #1161.

`predict_competing_risks_survival` evaluated its `time_grid` at every grid
time without the `t <= 0` origin special-case that single-cause
`predict_survival` applies. The default survival prediction grid spans
``lo = min(entry)`` to the training-time upper bound; for the conventional
``entry = 0`` survival design the first grid point is exactly ``t = 0``. At the
origin the time basis floors ``t`` to ``SURVIVAL_TIME_FLOOR = 1e-9`` before
taking ``log``, so the competing-risks loop returned a nonzero hazard there:
``S(0) = exp(-exp(eta)) < 1`` and ``CIF(0) > 0`` — biasing the Aalen-Johansen
CIF assembly and the whole overall-survival curve away from the mathematically
required origin values.

Single-cause `predict_survival` reports the correct ``S(0)=1, h(0)=0, H(0)=0``.
The fix mirrors that origin guard in the competing-risks grid loop so the two
predictors agree at ``t = 0``.
"""

import importlib
from typing import Any, cast

pytest = cast(Any, importlib.import_module("pytest"))

pytest.importorskip("gamfit._rust")

import numpy as np
import pandas as pd

import gamfit


def _make_competing_risks(n: int = 320, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.uniform(40.0, 75.0, n)
    x = (age - 55.0) / 10.0
    t1 = rng.exponential(scale=1.0 / np.exp(-3.0 + 0.25 * x), size=n)
    t2 = rng.exponential(scale=1.0 / np.exp(-3.2 - 0.20 * x), size=n)
    censor = rng.exponential(scale=22.0, size=n)
    exit_time = np.minimum.reduce([t1, t2, censor]) + 0.1
    event = np.where((t1 < t2) & (t1 < censor), 1.0, 0.0)
    event = np.where((t2 < t1) & (t2 < censor), 2.0, event)
    return pd.DataFrame(
        {
            "entry": np.zeros(n),
            "exit": exit_time,
            "event": event,
            "age": age,
        }
    )


def test_competing_risks_predict_has_unit_survival_at_origin() -> None:
    train = _make_competing_risks()
    model = gamfit.fit(
        train,
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="weibull",
        precision_hyperpriors={"cause_specific_survival_penalty_0": [2.0, 1.0]},
    )
    pred_rows = pd.DataFrame(
        {
            "entry": [0.0, 0.0, 0.0],
            "exit": [5.0, 10.0, 20.0],
            "event": [1, 1, 1],
            "age": [45.0, 55.0, 65.0],
        }
    )
    pred = model.predict(pred_rows)
    assert isinstance(pred, gamfit.CompetingRisksPrediction)

    times = np.asarray(pred.times, dtype=float)
    # The default grid for an entry==0 design starts exactly at the origin; that
    # is the cell the bug corrupted.
    assert times.size >= 1
    origin = int(np.argmin(times))
    assert times[origin] == 0.0, (
        f"default competing-risks grid should include t=0; got {times[:3]}"
    )

    overall_survival = np.asarray(pred.overall_survival, dtype=float)
    cif = np.asarray(pred.cif, dtype=float)
    hazard = np.asarray(pred.hazard, dtype=float)
    cumulative = np.asarray(pred.cumulative_hazard, dtype=float)

    # Every subject is alive at the time origin: S(0)=1, every cause-specific
    # CIF(0)=0, h(0)=0, H(0)=0. These must hold EXACTLY (the guard sets them to
    # the literal constants), not merely "close to" — the old floored-time path
    # returned ~1e-9..1e-8 deviations that anchored the whole CIF curve.
    np.testing.assert_array_equal(
        overall_survival[:, origin],
        np.ones(overall_survival.shape[0]),
    )
    np.testing.assert_array_equal(cif[:, origin], np.zeros(cif.shape[0]))
    np.testing.assert_array_equal(hazard[:, origin], np.zeros(hazard.shape[0]))
    np.testing.assert_array_equal(
        cumulative[:, origin], np.zeros(cumulative.shape[0])
    )

    # CIF + overall survival must still sum to 1 at the origin (degenerate: all
    # mass on survival).
    cause_count = len(pred.endpoint_names)
    n_rows = overall_survival.shape[0]
    assert cif.shape[0] == cause_count * n_rows
    cif_by_cause = cif.reshape(cause_count, n_rows, times.size)
    cif_sum_origin = cif_by_cause[:, :, origin].sum(axis=0)
    np.testing.assert_allclose(
        cif_sum_origin + overall_survival[:, origin], 1.0, atol=0.0
    )
