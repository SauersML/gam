from __future__ import annotations

import multiprocessing as mp
import queue
import time
from typing import Any

import pytest

pytest.importorskip("gamfit._rust")

import numpy as np
import pandas as pd

import gamfit


def make_weibull(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.uniform(40.0, 75.0, n)
    bmi = rng.uniform(18.0, 40.0, n)
    hba1c = rng.uniform(4.5, 9.0, n)
    eta = -2.0 + 0.04 * (age - 50.0) + 0.05 * (bmi - 25.0) + 0.4 * (hba1c - 6.0)
    shape = 1.5
    u = rng.uniform(1e-9, 1.0, n)
    t_lat = np.exp(-eta / shape) * (-np.log(u)) ** (1.0 / shape)
    t_lat *= 10.0
    c = rng.exponential(1.0 / (-np.log(0.5) / 25.0), n)
    c = np.minimum(c, 25.0)
    t_obs = np.minimum(t_lat, c)
    event = (t_lat <= c).astype(int)
    return pd.DataFrame(
        {
            "entry": np.zeros(n),
            "exit": t_obs,
            "event": event,
            "age": age,
            "bmi": bmi,
            "hba1c": hba1c,
        }
    )


def prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry": [0.0, 0.0, 0.0],
            "exit": [5.0, 10.0, 20.0],
            "event": [1, 1, 1],
            "age": [45.0, 55.0, 65.0],
            "bmi": [22.0, 28.0, 35.0],
            "hba1c": [5.2, 5.7, 6.5],
        }
    )


def make_competing_risks(n: int = 320, seed: int = 123) -> pd.DataFrame:
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


def test_survival_transformation_is_reachable_from_fit() -> None:
    model = gamfit.fit(
        make_weibull(260),
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="transformation",
    )
    pred = model.predict(prediction_rows())
    assert model.is_survival
    assert np.all(np.isfinite(np.asarray(pred.linear_predictor, dtype=float)))
    assert np.all(np.asarray(pred.survival, dtype=float) > 0.0)


def test_joint_competing_risks_survival_is_reachable_from_fit() -> None:
    train = make_competing_risks()
    validation = gamfit.validate_formula(
        train,
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="weibull",
    )
    assert validation["model_class"] == "competing risks survival"

    model = gamfit.fit(
        train,
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="weibull",
        precision_hyperpriors={"cause_specific_survival_penalty_0": [2.0, 1.0]},
    )
    pred = model.predict(prediction_rows()[["entry", "exit", "event", "age"]])
    assert isinstance(pred, gamfit.CompetingRisksPrediction)
    assert pred.endpoint_names == ("cause_1", "cause_2")
    assert pred.cif.shape[0] == 2
    assert pred.cif.shape[1] == 3
    assert pred.cif.shape[2] == pred.times.size
    assert np.all(np.isfinite(pred.cif))
    assert np.all((pred.cif >= 0.0) & (pred.cif <= 1.0))
    assert np.all((pred.overall_survival >= 0.0) & (pred.overall_survival <= 1.0))


def test_survival_location_scale_regressor_prediction_does_not_saturate() -> None:
    model = gamfit.fit(
        make_weibull(500),
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="location-scale",
    )
    pred = model.predict(prediction_rows())
    eta = np.asarray(pred.linear_predictor, dtype=float)
    survival = np.asarray(pred.survival, dtype=float)
    assert np.all(np.isfinite(eta))
    assert float(np.max(np.abs(eta))) < 50.0
    assert float(np.min(survival)) > 1.0e-12


def test_latent_survival_accepts_frailty_kwargs() -> None:
    validation = gamfit.validate_formula(
        make_weibull(120),
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="latent",
        baseline_target="weibull",
        baseline_shape=1.5,
        baseline_scale=10.0,
        frailty_kind="hazard-multiplier",
        hazard_loading="full",
    )
    assert validation["model_class"] == "latent survival"
    assert validation.supported_by_python is True


def _fit_marginal_slope_worker(result_queue: mp.Queue[Any]) -> None:
    try:
        gamfit.fit(
            make_weibull(220),
            "Surv(entry, exit, event) ~ bmi + hba1c",
            survival_likelihood="marginal-slope",
            z_column="age",
            logslope_formula="bmi + hba1c",
        )
    except BaseException as exc:  # pragma: no cover - child process reporting
        result_queue.put(("error", type(exc).__name__, str(exc)))
    else:
        result_queue.put(("ok",))


def test_survival_marginal_slope_fit_returns() -> None:
    result_queue: mp.Queue[Any] = mp.Queue()
    proc = mp.Process(target=_fit_marginal_slope_worker, args=(result_queue,))
    start = time.monotonic()
    proc.start()
    proc.join(45.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        pytest.fail("survival marginal-slope fit did not return within 45 seconds")
    assert proc.exitcode == 0
    try:
        result = result_queue.get_nowait()
    except queue.Empty as exc:
        raise AssertionError("survival marginal-slope worker returned no result") from exc
    assert result == ("ok",), result
    assert time.monotonic() - start < 45.0
