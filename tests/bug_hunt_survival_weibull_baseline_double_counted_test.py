"""Bug hunt: parametric-Weibull survival (``--survival-likelihood weibull``)
double-counts the baseline hazard, so its survival predictions are grossly wrong.

A parametric Weibull survival model with a linear predictor ``x`` has
log-cumulative-hazard

    log H(t | x) = k·log(t) − k·log(λ) + β·x           (proportional hazards)

so, at a fixed covariate value, ``eta = log H(t)`` is *linear in log t with slope
exactly the Weibull shape ``k``*, and the survival curve is
``S(t) = exp(−exp(eta))``.

Fitting a lightly-censored Weibull sample (true shape ``k = 1.5``, scale
``λ = 10``) with ``--survival-likelihood weibull`` and predicting at the baseline
(``x = 0``) gives, instead:

* ``--mode map`` : ``eta`` has slope ≈ ``3.06`` in log t — **exactly 2·k** — so
  the survival curve decays twice as fast as the truth (e.g. ``S(5)`` comes back
  ``0.018`` where the truth is ``0.702``).
* the default ``--mode posterior-mean`` : ``S(t) ≈ 0.500`` for **every** time
  (the doubled, unidentified time-column posterior variance is enormous, so
  ``E[exp(−exp(eta + N(0, se²)))] → ½``), i.e. a completely degenerate curve.

The *same data* fit with ``--survival-likelihood transformation`` recovers the
truth to <0.01 in survival (``eta`` slope ≈ ``1.55 ≈ k``), which proves the data,
the harness, and the survival predict path are otherwise sound — the defect is
specific to the Weibull-mode baseline.

Root cause (read, not patched): the Weibull single-cause fit runs the baseline on
a *linear* time basis ``[1, log t]`` (so the whole baseline lives in
``beta_time``, with ``beta_time[1] ≈ k``), and the save path re-encodes that same
baseline a SECOND time as a *parametric* Weibull ``baseline_cfg``
(``fitted_weibull_baseline_from_linear_time_beta`` in
``crates/gam-cli/src/main/run_survival.rs`` ~lines 75-91). At predict time the
Royston-Parmar branch rebuilds a parametric offset ``shape·(log t − log scale)``
from that saved ``baseline_cfg``
(``crates/gam-models/src/survival/construction.rs`` ~lines 2736-2739) AND ALSO
applies the linear time design with ``beta_time`` on top — so ``k·log t`` is
counted twice and ``eta`` slope is ``2·k``. Unlike the transformation /
location-scale / latent paths, the Weibull path is excluded from the anchor
re-centering in ``crates/gam-cli/src/main/run_predict.rs`` (~lines 1059-1075),
which is why only Weibull mode double-counts.

The test fits both modes on the same Weibull sample and asserts the Weibull-mode
baseline is recovered (slope ≈ k, survival ≈ truth), using the transformation fit
as a passing control. It currently FAILS (slope ≈ 2·k, survival wildly off) and
will PASS once the Weibull baseline stops being applied twice.

Related: #2112 (parametric-AFT location-scale MLE convergence), #1842/#1847
(other survival-mode predict/variant defects) — this is a distinct baseline
double-count in the Weibull likelihood mode.

The test drives the ``gam`` CLI (on $PATH); the Python ``gamfit`` wheel is not
required.
"""

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

GAM = shutil.which("gam")

TRUE_SHAPE = 1.5
TRUE_SCALE = 10.0
TRUE_BETA = 0.8
GRID_TIMES = [2.0, 5.0, 10.0, 20.0]


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))


def _make_data(wd):
    rng = np.random.default_rng(0)
    n = 2000
    k, lam, beta = TRUE_SHAPE, TRUE_SCALE, TRUE_BETA
    x = rng.normal(0.0, 1.0, n)
    # Weibull proportional-hazards event times: H(t|x) = (t/lam)^k * exp(beta*x).
    # Inverse-CDF draw with the covariate folded into the scale.
    u = rng.random(n)
    t_event = (lam * np.exp(-beta * x / k)) * (-np.log(u)) ** (1.0 / k)
    admin = 40.0  # light administrative censoring (~2%)
    exit_time = np.minimum(t_event, admin)
    event = (t_event <= admin).astype(int)

    train = wd / "surv_train.csv"
    _write(
        train,
        ["t0", "t1", "event", "x"],
        [(0.0, exit_time[i], int(event[i]), x[i]) for i in range(n)],
    )
    grid = wd / "surv_grid.csv"
    _write(grid, ["t0", "t1", "event", "x"], [(0.0, t, 1, 0.0) for t in GRID_TIMES])
    return train, grid


def _fit(train, likelihood, out):
    r = subprocess.run(
        [
            GAM,
            "fit",
            str(train),
            "Surv(t0,t1,event) ~ x",
            "--survival-likelihood",
            likelihood,
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"fit ({likelihood}) failed: {r.stderr}"


def _predict(model, grid, out, mode):
    r = subprocess.run(
        [GAM, "predict", str(model), str(grid), "--mode", mode, "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"predict ({mode}) failed: {r.stderr}"
    rows = list(csv.DictReader(open(out)))
    return {k: np.array([float(x[k]) for x in rows]) for k in rows[0]}


def _true_survival(times):
    t = np.asarray(times, dtype=float)
    return np.exp(-((t / TRUE_SCALE) ** TRUE_SHAPE))


@pytest.mark.skipif(GAM is None, reason="gam CLI not on PATH")
def test_weibull_survival_baseline_is_not_double_counted():
    log_t = np.log(np.asarray(GRID_TIMES))
    true_s = _true_survival(GRID_TIMES)

    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        train, grid = _make_data(wd)

        _fit(train, "transformation", wd / "m_tf.json")
        _fit(train, "weibull", wd / "m_wb.json")

        tf_map = _predict(wd / "m_tf.json", grid, wd / "p_tf.csv", "map")
        wb_map = _predict(wd / "m_wb.json", grid, wd / "p_wb.csv", "map")
        wb_pm = _predict(wd / "m_wb.json", grid, wd / "p_wb_pm.csv", "posterior-mean")

    # ── Control: transformation mode recovers the true Weibull baseline. This
    # proves the data, the CLI, and the survival predict path are sound, so any
    # failure below is specific to the Weibull likelihood mode. ──
    tf_slope = float(np.polyfit(log_t, tf_map["eta"], 1)[0])
    assert abs(tf_slope - TRUE_SHAPE) < 0.4, (
        f"control failed: transformation-mode log-cumulative-hazard slope "
        f"{tf_slope:.3f} should recover the true Weibull shape {TRUE_SHAPE}; the "
        f"test harness assumption is violated."
    )
    assert np.max(np.abs(tf_map["survival_prob"] - true_s)) < 0.05, (
        f"control failed: transformation-mode survival "
        f"{tf_map['survival_prob']} should match the truth {true_s}."
    )

    # ── The bug: Weibull-mode `eta = log H(t)` must be linear in log t with
    # slope equal to the true shape k. It comes back ≈ 2·k because the baseline
    # is applied twice (linear time design + re-encoded parametric offset). ──
    wb_slope = float(np.polyfit(log_t, wb_map["eta"], 1)[0])
    assert abs(wb_slope - TRUE_SHAPE) < 0.4, (
        f"Weibull-mode survival baseline is DOUBLE-COUNTED: the fitted "
        f"log-cumulative-hazard slope in log t is {wb_slope:.3f}, but a Weibull "
        f"model with shape k={TRUE_SHAPE} must have slope k. The observed slope "
        f"is ~2·k ({2 * TRUE_SHAPE}), i.e. the baseline k·log(t) is added twice "
        f"(linear time design + re-encoded parametric Weibull offset). The "
        f"transformation fit on identical data recovers slope {tf_slope:.3f}."
    )

    # ── Consequence: the map-mode survival curve is therefore wildly wrong. ──
    assert np.max(np.abs(wb_map["survival_prob"] - true_s)) < 0.1, (
        f"Weibull-mode (map) survival {wb_map['survival_prob']} is grossly wrong "
        f"vs the truth {true_s} because the baseline is double-counted."
    )

    # ── Second symptom: the DEFAULT posterior-mean mode collapses the survival
    # curve to ≈ 0.5 at every time (the doubled, unidentified time column has an
    # enormous posterior variance). A valid model must not report a flat 0.5. ──
    assert np.max(np.abs(wb_pm["survival_prob"] - 0.5)) > 0.1, (
        f"Weibull-mode default (posterior-mean) survival collapsed to a flat "
        f"~0.5 at every time: {wb_pm['survival_prob']} (truth {true_s}). A valid "
        f"fit must produce a non-degenerate, time-varying survival curve."
    )
