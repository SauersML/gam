"""Bug hunt: ``gam predict`` on a conditional transformation-normal (CTM) model
returns the probability-integral transform ``h(y|x)`` of the *supplied response*
as both the ``linear_predictor`` and the ``mean`` column — not a response-scale
prediction ``E[Y|x]``.

A transformation-normal model fits ``h(Y|x) ~ N(0,1)`` with ``h`` monotone in
``y`` and covariate effects in ``x``. ``gam predict`` is documented as "Predict
on a new dataset"; for every other family the ``mean`` column is the conditional
mean ``E[Y|x]`` computed from the covariates. A prediction is therefore a
function of the covariates alone: at a fixed ``x`` it cannot depend on which
outcome value ``y`` you hand it.

The CTM predictor breaks this. ``crates/gam-predict/src/transformation_normal.rs``
(lines 3-28) documents and implements:

    "The PIT-transformed values h(y|x) are precomputed in
     `build_predict_input_for_model` and stored in the PredictInput offset.
     This predictor passes them through as the prediction: eta = h, mean = h."

        let h = input.offset.clone();
        Ok(LinearState { eta: h.clone(), mean: h, ... })

So the reported ``mean`` is ``h(y|x)`` (a per-row standard-normal residual), which
(a) *requires the response column at prediction time* and (b) varies monotonically
with ``y`` at fixed ``x``.

Reproduction (this test): fit a CTM with a genuinely monotone conditional mean
``E[Y|x] = 2 + 0.9x``, then predict a 5-row frame holding ``x = 1`` fixed while
``y`` sweeps ``-2 … 6``. The conditional mean is constant across those rows, but
the engine returns:

    linear_predictor,mean
    -7.034483825305,-7.034483825305
    -5.796814382494,-5.796814382494
    -1.789977907165,-1.789977907165
     2.152049428025, 2.152049428025
     7.034486910051, 7.034486910051

— a 14-unit monotone sweep in ``y`` at fixed ``x``. (Predicting on a
covariate-only frame is rejected outright with "requested column 'y' not found",
the same root cause from a different angle.)

This test asserts the unimpeachable invariant: the predicted ``mean`` at a fixed
covariate value must be (nearly) constant across different supplied responses,
because ``E[Y|x]`` is a function of ``x`` alone. It currently fails (the means
span ~14 units); once predict returns a covariate-only response-scale prediction
it passes, with no further edits.

Driven through the ``gam`` CLI (the buildable front-end), so it does not depend
on the gamfit wheel.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

# Prefer the workspace release binary (the suite's convention); fall back to a
# `gam` on PATH so the check runs against the installed engine too.
_REPO_BIN = os.path.join(os.path.dirname(__file__), "..", "target", "release", "gam")
GAM_BIN = _REPO_BIN if os.path.exists(_REPO_BIN) else (shutil.which("gam") or _REPO_BIN)

pytestmark = pytest.mark.skipif(
    not os.path.exists(GAM_BIN),
    reason="`gam` CLI binary not available (neither target/release/gam nor PATH)",
)


def _write_csv(path: str, cols: dict[str, np.ndarray]) -> None:
    keys = list(cols)
    n = len(cols[keys[0]])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([cols[k][i] for k in keys])


def _read_pred_mean(path: str) -> np.ndarray:
    with open(path) as f:
        r = csv.DictReader(f)
        assert "mean" in (r.fieldnames or []), f"no 'mean' column in {r.fieldnames}"
        return np.array([float(row["mean"]) for row in r])


def test_ctm_predicted_mean_is_function_of_covariates_only() -> None:
    rng = np.random.default_rng(3)
    n = 1500
    x = rng.uniform(-2.0, 2.0, n)
    # Genuinely monotone conditional mean E[Y|x] = 2 + 0.9 x.
    y = 2.0 + 0.9 * x + rng.normal(0.0, 0.5, n)

    with tempfile.TemporaryDirectory() as d:
        train = os.path.join(d, "train.csv")
        grid = os.path.join(d, "grid.csv")
        model = os.path.join(d, "ctm.gam")
        pred = os.path.join(d, "pred.csv")

        _write_csv(train, {"y": y, "x": x})

        fit = subprocess.run(
            [GAM_BIN, "fit", train, "y ~ smooth(x)", "--transformation-normal",
             "--out", model],
            capture_output=True, text=True,
        )
        assert fit.returncode == 0, f"fit failed: {fit.stderr[-2000:]}"

        # Five rows at the SAME covariate x = 1.0, sweeping the response y.
        grid_y = np.array([-2.0, 0.0, 2.0, 4.0, 6.0])
        grid_x = np.full_like(grid_y, 1.0)
        _write_csv(grid, {"y": grid_y, "x": grid_x})

        out = subprocess.run(
            [GAM_BIN, "predict", model, grid, "--out", pred],
            capture_output=True, text=True,
        )
        assert out.returncode == 0, f"predict failed: {out.stderr[-2000:]}"

        means = _read_pred_mean(pred)
        assert means.size == grid_y.size

        spread = float(means.max() - means.min())
        # E[Y|x=1] is one number; the five predictions must coincide. Allow a
        # generous absolute slack for numerical noise — the bug produces a
        # ~14-unit monotone sweep in y, orders of magnitude above any tolerance.
        assert spread < 1e-3, (
            "transformation-normal predict() returns a per-row value that depends "
            f"on the supplied response y: predicted mean at fixed x=1 spans {spread:.4f} "
            f"across y in {list(grid_y)} (means={list(means)}). A prediction must be a "
            "function of the covariates only (E[Y|x] is constant at fixed x); the CTM "
            "predictor instead passes through the PIT residual h(y|x) as the 'mean'."
        )
