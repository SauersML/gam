"""Bug hunt: ``gam generate`` on a conditional transformation-normal (CTM) model
produces synthetic responses on the *latent* N(0,1) scale (not the response
scale), requires the response column, and yields a conditional mean that moves
the **wrong way** with the covariate.

``gam generate`` is documented as "Draw synthetic responses from the fitted model
for given covariates". For a CTM fit to data with a strictly *increasing*
conditional mean ``E[Y|x] = 2 + 0.9x``, the per-row mean of the synthetic draws
must therefore *increase* with ``x`` and sit on the response scale.

Instead, generation goes through the same broken CTM plug-in path as prediction:
``run_generate_unified`` (crates/gam-cli/src/main/run_sample_generate_report.rs)
asks the predictor for the response-scale plug-in mean via
``predict_plugin_response``, but the ``TransformationNormalPredictor`` returns the
PIT/latent value ``h(y|x)`` as that "mean" (see
``crates/gam-predict/src/transformation_normal.rs:19``, ``eta = mean = h``; cf.
the prediction-side report in the sibling issue). The observation sampler then
draws ``N(h(y|x), residual_sd)`` on the latent scale, so:

  * generation *requires* the outcome column ``y`` (it errors
    "requested column 'y' not found" on a covariate-only frame), and
  * the draws are centered on ``h(y|x)`` — sd ≈ 1 (the latent scale, not the
    response residual sd ≈ 0.5), with a conditional mean *decreasing* in ``x``:

        x=-1: gen mean=-2.40 sd=1.02   true E[Y|x]=1.1
        x= 0: gen mean=-4.02 sd=1.02   true E[Y|x]=2.0
        x= 1: gen mean=-5.83 sd=0.99   true E[Y|x]=2.9

Reproduction (this test): fit a CTM with ``E[Y|x] = 2 + 0.9x`` (strictly
increasing), then generate many draws at ``x ∈ {-1, 0, 1}`` and check that the
per-row mean of the draws increases with ``x``. It currently fails (the means
strictly *decrease*); once generation produces response-scale draws it passes,
with no further edits.

Related to the prediction-side report (CTM ``predict`` returns ``h(y|x)`` as the
mean) — same root cause, different command/path. Driven through the ``gam`` CLI,
so it does not depend on the gamfit wheel.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

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


def _read_matrix(path: str) -> np.ndarray:
    rows = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            rows.append([float(v) for v in row])
    return np.array(rows)


def test_ctm_generate_conditional_mean_increases_with_covariate() -> None:
    rng = np.random.default_rng(3)
    n = 1500
    x = rng.uniform(-2.0, 2.0, n)
    y = 2.0 + 0.9 * x + rng.normal(0.0, 0.5, n)  # strictly increasing E[Y|x]

    with tempfile.TemporaryDirectory() as d:
        train = os.path.join(d, "train.csv")
        grid = os.path.join(d, "grid.csv")
        model = os.path.join(d, "ctm.gam")
        gen = os.path.join(d, "gen.csv")

        _write_csv(train, {"y": y, "x": x})

        # `s(x, k=6)` (a modest rank-6 marginal) keeps the CTM transformation
        # basis well-conditioned for n=1500; the default `smooth(x)` auto-knots
        # to a rank that blows the joint transformation basis up to ~168 params,
        # which is over-parameterised for this sample and the inner monotone
        # solve correctly refuses to certify a non-converged fit. The generate
        # contract under test (response-scale draws whose mean increases with x)
        # is independent of the marginal rank — this mirrors the convergent
        # basis used by the sibling predict-side regression test.
        fit = subprocess.run(
            [GAM_BIN, "fit", train, "y ~ s(x, k=6)", "--transformation-normal",
             "--out", model],
            capture_output=True, text=True,
        )
        assert fit.returncode == 0, f"fit failed: {fit.stderr[-2000:]}"

        # Three covariate rows with increasing x. (A placeholder `y` column is
        # supplied only because the broken path demands it; a correct
        # covariate-only generator would not need it and would still pass.)
        grid_x = np.array([-1.0, 0.0, 1.0])
        _write_csv(grid, {"y": np.zeros_like(grid_x), "x": grid_x})

        out = subprocess.run(
            [GAM_BIN, "generate", model, grid, "--n-draws", "3000", "--seed", "2",
             "--out", gen],
            capture_output=True, text=True,
        )
        assert out.returncode == 0, f"generate failed: {out.stderr[-2000:]}"

        draws = _read_matrix(gen)  # shape (n_rows, n_draws)
        assert draws.shape[0] == grid_x.size
        row_means = draws.mean(axis=1)

        # E[Y|x] = 2 + 0.9x is strictly increasing, so the synthetic draws'
        # per-row mean must increase with x.
        assert row_means[0] < row_means[1] < row_means[2], (
            "transformation-normal generate() produces draws whose conditional mean "
            f"does not increase with x: row means {list(row_means)} at x={list(grid_x)} "
            "for a model fit to strictly increasing E[Y|x]=2+0.9x. The CTM generator "
            "draws N(h(y|x), sd) on the latent scale instead of response-scale Y, so the "
            "mean moves the wrong way with the covariate."
        )
