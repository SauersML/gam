from __future__ import annotations

import multiprocessing as mp
import queue
import time
import typing

import numpy as np
import pandas as pd

import gamfit
import pytest

# #1512: this fit exceeds the standard Python-API CI runner budget (>60s in
# triage), so it is tagged slow and excluded from the directory-level
# `-m "not slow"` CI step while still being collected (run by a bare pytest).
pytestmark = pytest.mark.slow


HARD_N = 195_780
HARD_TIMEOUT_SECONDS = 420.0


def _large_scale_survival_marginal_slope_frame(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pc1 = np.clip(rng.normal(0.0243522, 0.042, n), -0.339466, 0.118391)
    pc2 = np.clip(rng.normal(0.0807963, 0.030, n), -0.186896, 0.144561)
    pc3 = np.clip(rng.normal(-0.00891745, 0.036, n), -0.313383, 0.0562003)
    sex = (rng.uniform(0.0, 1.0, n) < 0.391184).astype(np.int64)

    prs_z = rng.normal(0.0, 1.0, n)
    prs_z = (prs_z - prs_z.mean()) / prs_z.std(ddof=0)

    entry_age = np.clip(rng.normal(45.0827, 18.0, n), 1.54689, 121.963)
    followup = rng.gamma(shape=1.7, scale=4.4, size=n) + 0.05
    exit_age = np.minimum(entry_age + followup, 122.47)
    exit_age = np.maximum(exit_age, entry_age + 1.0e-3)

    event_score = (
        0.34 * prs_z
        + 0.15 * sex
        + 2.4 * (pc1 - pc1.mean())
        - 1.6 * (pc2 - pc2.mean())
        + 1.9 * (pc3 - pc3.mean())
        + rng.normal(0.0, 1.0, n)
    )
    event = (event_score >= np.median(event_score)).astype(np.int64)

    return pd.DataFrame(
        {
            "entry_age": entry_age,
            "exit_age": exit_age,
            "event": event,
            "sex": sex,
            "prs_z": prs_z,
            "PC1": pc1,
            "PC2": pc2,
            "PC3": pc3,
        }
    )


def _fit_large_scale_survival_marginal_slope_worker(result_queue: typing.Any) -> None:
    try:
        df = _large_scale_survival_marginal_slope_frame(seed=0xA0B10B, n=HARD_N)
        duchon = "duchon(PC1, PC2, PC3, centers=10, order=1)"
        started = time.monotonic()
        model = gamfit.fit(
            df,
            f"Surv(entry_age, exit_age, event) ~ {duchon} + sex",
            survival_likelihood="marginal-slope",
            z_column="prs_z",
            logslope_formula=duchon,
        )
        elapsed = time.monotonic() - started
        pred = model.predict(df.iloc[:16].copy())
        survival = np.asarray(pred.survival, dtype=float)
        result_queue.put(
            (
                "ok",
                elapsed,
                survival.shape,
                bool(np.all(np.isfinite(survival))),
                float(np.min(survival)),
                float(np.max(survival)),
            )
        )
    except BaseException as exc:
        result_queue.put(("error", type(exc).__name__, str(exc)))


def test_survival_marginal_slope_large_scale_startup_seeds_do_not_all_reject() -> None:
    """Hard red repro for the large-scale survival marginal-slope startup failure.

    The production failure is not a small scalar-kernel bug. It happens only
    after the Python FFI builds the full survival marginal-slope model shape,
    then every outer seed is rejected because the exact joint inner solve exits
    the Newton path before convergence.
    """
    if not gamfit.build_info().get("available"):
        raise AssertionError("rust extension is required for the hard large-scale startup repro")

    result_queue = mp.Queue()
    proc = mp.Process(target=_fit_large_scale_survival_marginal_slope_worker, args=(result_queue,))
    proc.start()
    proc.join(HARD_TIMEOUT_SECONDS)
    if proc.is_alive():
        proc.terminate()
        proc.join(10.0)
        raise AssertionError(
            f"survival marginal-slope hard large-scale startup repro did not return within "
            f"{HARD_TIMEOUT_SECONDS:.0f}s"
        )

    assert proc.exitcode == 0
    try:
        result = result_queue.get_nowait()
    except queue.Empty as exc:
        raise AssertionError("survival marginal-slope hard worker returned no result") from exc

    assert result[0] == "ok", result
    _, elapsed, shape, finite, survival_min, survival_max = result
    assert elapsed < HARD_TIMEOUT_SECONDS
    assert shape[0] == 16
    assert finite
    assert 0.0 < survival_min < survival_max < 1.0
