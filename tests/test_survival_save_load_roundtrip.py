"""Round-trip integration tests: fit → save → load → predict → compare.

These tests guard the class of bug where the pyffi save path for a
particular survival likelihood mode forgets to populate a field that
load+predict needs. The bug class is invisible to fit-only and
predict-only tests because the in-memory model carries the full state;
only a roundtrip through disk exposes the omission.

A regression here would catch e.g. the survival-marginal-slope payload
omitting ``survival_time_basis`` / ``survival_time_anchor`` — the exact
shape of the bug fixed in
``crates/gam-pyffi`` commit fa7f7b6c. With that fix in place,
``FittedModel::save_to_path`` runs ``validate_for_persistence`` before
writing bytes, so a regression would fail at the ``model.save`` step
rather than the ``gamfit.load`` step — either way, this test is the
canary.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import tempfile
import time
from typing import Any

import pytest

pytest.importorskip("gamfit._rust")

import numpy as np
import pandas as pd

import gamfit


# ── synthetic-data helpers ────────────────────────────────────────────
# Mirror the fixtures used by tests/test_survival_api_regressions.py so
# the roundtrip tests exercise the same shapes as the existing fit-only
# tests, just with an added save+load+predict step.

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
    # entry > 0 so baseline targets that require positive ages (transformation
    # / weibull) accept the rows; the default time grid for predict is built
    # from these entry/exit columns.
    return pd.DataFrame(
        {
            "entry": [1.0, 1.0, 1.0],
            "exit": [5.0, 10.0, 20.0],
            "event": [1, 1, 1],
            "age": [45.0, 55.0, 65.0],
            "bmi": [22.0, 28.0, 35.0],
            "hba1c": [5.2, 5.7, 6.5],
        }
    )


def _roundtrip_survival(model: Any, sample: pd.DataFrame) -> None:
    """Save → load → predict → assert survival surface matches in-memory.

    Compares the ``SurvivalPrediction.survival`` matrix (shape
    ``(n_rows, n_grid_points)`` on gamfit's default time grid derived from
    the prediction frame). The surface depends on every field the predict
    path reads from the saved payload (time basis, anchor, baseline
    target/parameters, link wiggle, …), so a missing field manifests as
    either a load error or a numerical mismatch.
    """
    pred_inmem = model.predict(sample)
    surv_inmem = np.asarray(pred_inmem.survival, dtype=float)
    assert surv_inmem.shape[0] == len(sample), surv_inmem.shape
    assert np.all(np.isfinite(surv_inmem))
    assert np.all((surv_inmem > 0.0) & (surv_inmem <= 1.0))

    fd, tmp_path = tempfile.mkstemp(suffix=".gamfit")
    os.close(fd)
    try:
        model.save(tmp_path)
        reloaded = gamfit.load(tmp_path)
        pred_disk = reloaded.predict(sample)
        surv_disk = np.asarray(pred_disk.survival, dtype=float)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    np.testing.assert_allclose(surv_inmem, surv_disk, atol=1e-9, rtol=1e-9)


# ── transformation + location-scale: run inline (cheap fits) ──────────

def test_survival_transformation_save_load_predict_roundtrips() -> None:
    model = gamfit.fit(
        make_weibull(260),
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="transformation",
    )
    _roundtrip_survival(model, prediction_rows())


def test_survival_location_scale_save_load_predict_roundtrips() -> None:
    model = gamfit.fit(
        make_weibull(500),
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="location-scale",
    )
    _roundtrip_survival(model, prediction_rows())


# ── marginal-slope: subprocess + time budget (matches the existing
#   fit-only regression test pattern) ─────────────────────────────────

def _marginal_slope_roundtrip_worker(result_queue: Any) -> None:
    try:
        import numpy as _np
        import tempfile as _tf
        df = make_weibull(220)
        model = gamfit.fit(
            df,
            "Surv(entry, exit, event) ~ bmi + hba1c",
            survival_likelihood="marginal-slope",
            z_column="age",
            logslope_formula="bmi + hba1c",
        )
        sample = prediction_rows()
        pred_inmem = model.predict(sample)
        surv_inmem = _np.asarray(pred_inmem.survival, dtype=float)

        fd, tmp_path = _tf.mkstemp(suffix=".gamfit")
        os.close(fd)
        try:
            model.save(tmp_path)
            reloaded = gamfit.load(tmp_path)
            pred_disk = reloaded.predict(sample)
            surv_disk = _np.asarray(pred_disk.survival, dtype=float)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if not _np.allclose(surv_inmem, surv_disk, atol=1e-9, rtol=1e-9):
            result_queue.put(
                (
                    "error",
                    "RoundtripMismatch",
                    "survival surface differs between in-memory and reloaded "
                    "model",
                )
            )
            return
    except BaseException as exc:  # pragma: no cover - child process reporting
        result_queue.put(("error", type(exc).__name__, str(exc)))
    else:
        result_queue.put(("ok",))


def test_survival_marginal_slope_save_load_predict_roundtrips() -> None:
    result_queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_marginal_slope_roundtrip_worker, args=(result_queue,)
    )
    start = time.monotonic()
    proc.start()
    proc.join(60.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        pytest.fail(
            "survival marginal-slope fit+save+load+predict did not return "
            "within 60 seconds"
        )
    assert proc.exitcode == 0
    try:
        result = result_queue.get_nowait()
    except queue.Empty as exc:
        raise AssertionError(
            "survival marginal-slope roundtrip worker returned no result"
        ) from exc
    assert result == ("ok",), result
    assert time.monotonic() - start < 60.0
