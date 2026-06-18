"""#1270 — `matern(x1, x2)` must fit an ordinary 2-D smooth-regression surface
like its radial/tensor siblings, not deterministically abort.

Root cause: the single-spatial-term n-free penalty re-key (the #1033 κ-optimizer
fast path) rebuilt the Matérn penalty surface S(ψ) through the kernel
double-penalty path (1 block), but the realized Matérn design ALWAYS uses the
canonical {mass, tension, stiffness} operator triplet (3 blocks for ν=5/2, d=2).
Every skip-path eval staged a 1-block surface against the 3-block frozen design,
tripped "penalty topology is not ψ-stable", cleared the stage, then hard-errored
"no exact S(psi) was staged" — aborting before any optimizer iteration ran.

The fix routes the re-key and the design builder through one shared
operator-triplet builder, so the block count is ψ-stable by construction.
"""

import numpy as np
import pandas as pd

import gamfit


def _gauss_bump_dataset(seed, n=400):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    truth = np.exp(-((x1 - 0.5) ** 2 + (x2 - 0.5) ** 2) * 6.0)
    y = truth + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    return df, truth


def _fit_recovery(term, seed):
    df, truth = _gauss_bump_dataset(seed)
    model = gamfit.fit(df, f"y ~ {term}")
    pred = np.asarray(model.predict(df))
    return float(np.corrcoef(pred, truth)[0, 1])


def test_matern_2d_smooth_fits_ordinary_surface():
    # Before the #1270 fix this raised IntegrationError on every seed. After the
    # fix it fits and recovers the surface like the sibling radial smooths.
    corrs = [_fit_recovery("matern(x1, x2)", seed) for seed in range(6)]
    assert min(corrs) > 0.9, corrs


def test_matern_2d_smooth_fits_at_larger_basis():
    # The abort was independent of k; verify a richer basis also fits and
    # recovers (it actually recovers better, ruling out a k-specific escape).
    corr = _fit_recovery("matern(x1, x2, k=20)", 5)
    assert corr > 0.95, corr


def test_matern_2d_smooth_fits_across_nu():
    # The operator-triplet topology (and its ψ-stability) depends on ν via the
    # Sobolev order m = ν + d/2; every supported ν for d=2 must fit.
    for nu in ("3/2", "5/2", "7/2"):
        corr = _fit_recovery(f"matern(x1, x2, nu={nu})", 5)
        assert corr > 0.9, (nu, corr)


def test_duchon_2d_control_fits_ordinary_surface():
    # Control: the sibling radial smooth was never affected and proves the data
    # is an easy fit.
    corr = _fit_recovery("duchon(x1, x2)", 5)
    assert corr > 0.9, corr
