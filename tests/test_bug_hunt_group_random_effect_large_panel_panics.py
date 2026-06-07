"""Regression test: a ``group()`` random-effect fit must not panic on large panels.

Fitting a Gaussian random-intercept model ``y ~ group(site)`` aborts inside the
Rust boundary as soon as the coefficient count reaches ``p >= 256`` (i.e. >= 255
groups), or whenever ``n_obs * p`` exceeds the small-problem budget. A 255-group
panel is an entirely ordinary mixed-model size, so this forecloses a common use
case.

Observed (with 300 groups):

    GamError: fit_table panicked inside Rust boundary: Assertion failed at
    faer-0.24.0/.../cholesky/llt/solve.rs:20
    Assertion failed: rhs.nrows() == n
    - rhs.nrows() = 301
    - n = 0

Root cause (chain), from reading the source:

* ``select_reml_geometry`` (``src/solver/reml/inner_strategy.rs:106``) routes to
  the dense backend only when ``p < 256 && n_obs * p < SMALL_NP_DENSE_BUDGET``
  (4_000_000). Otherwise it picks ``RemlGeometry::SparseExactSpd``. Random-
  intercept Gram matrices are extremely sparse, so the sparse-exact path fires
  readily for moderate group counts.
* The sparse-exact bundle stores a *placeholder* empty dense Hessian
  (``Array2::zeros((0, 0))``); ``RemlState::objective_innerhessian``
  (``src/solver/reml/runtime.rs:4735``) returns that 0x0 matrix verbatim, with
  no geometry awareness.
* ``compute_smoothing_correction`` (``src/solver/estimate.rs:1379``) feeds that
  0x0 matrix into ``h_trans.cholesky(Side::Lower)`` (line 1424). faer happily
  returns an ``n = 0`` factor for an empty matrix, so the ``Err`` guard never
  fires, and the subsequent ``ift_dbeta_drho_from_solver`` closure calls
  ``h_chol.solvevec(rhs)`` (line 1468) with ``rhs`` of length ``p``. faer's
  Cholesky solve then asserts ``rhs.nrows() == n`` (n = 0), aborting the fit.

The crash boundary is exact: 254 groups (p = 255) fits; 255 groups (p = 256)
panics. The fix (materialize a real Hessian for the sparse-exact geometry, or
detect the placeholder and skip / route the correction) is not prescribed here;
this test only asserts that a large random-intercept model fits and recovers
the group means it was generated from.

Related: #792, #674 (unseen-level prediction for ``group()``); this is a
distinct failure in the REML sparse-exact geometry path, not the predict path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def _make_panel(n_groups: int, per_group: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    true_effect = rng.normal(0.0, 2.0, n_groups)
    intercept = 5.0
    rows = []
    for i in range(n_groups):
        for _ in range(per_group):
            rows.append((f"g{i}", intercept + true_effect[i] + rng.normal(0.0, 1.0)))
    df = pd.DataFrame(rows, columns=["site", "y"])
    true_group_mean = intercept + true_effect
    return df, true_group_mean


def test_group_random_intercept_fits_large_panel() -> None:
    # 300 groups => p = 301 >= 256: pre-fix this aborts inside the Rust boundary
    # at the empty (0x0) Hessian Cholesky solve. The same model at 254 groups
    # fits cleanly, so the only difference is the geometry-selection branch.
    n_groups = 300
    df, true_group_mean = _make_panel(n_groups, per_group=20, seed=0)

    model = gamfit.fit(df, "y ~ group(site)", family="gaussian")

    # The fit must produce a usable model with a finite marginal-likelihood score.
    assert np.isfinite(model.evidence)

    # And it must actually recover the group structure it was generated from:
    # one prediction per group should track the true per-group mean closely.
    one_per_group = pd.DataFrame({"site": [f"g{i}" for i in range(n_groups)]})
    pred = np.asarray(model.predict(one_per_group)).ravel()
    assert pred.shape == (n_groups,)
    assert np.all(np.isfinite(pred))

    corr = np.corrcoef(pred, true_group_mean)[0, 1]
    # The identical model at 254 groups achieves corr ~= 0.99; require a clear
    # recovery (shrinkage keeps this below 1.0) well above chance.
    assert corr > 0.9, f"random-intercept BLUPs failed to recover group means: corr={corr}"
