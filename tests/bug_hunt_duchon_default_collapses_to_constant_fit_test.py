"""Default ``duchon(x1, x2)`` must not collapse to a constant fit (issue #1355).

The default 2-D Duchon radial smooth ``y ~ duchon(x1, x2)`` intermittently
collapsed to a degenerate constant fit (EDF≈1, prediction std≈1e-4, in-sample
R²≈0 against the noise-free truth) on perfectly ordinary, strong-signal data.
The collapse was *data-realization dependent and deterministic per seed*: at
``n=300`` on a smooth low-noise surface, seeds 0, 2, 4, 7 collapsed while their
neighbours fit cleanly (R²≈0.998).

Root cause (now fixed): the default Duchon ``Primary`` penalty was the raw
native reproducing-norm knot-Gram ``α²·Zᵀ K_CC Z`` — a cliff-less Mercer
spectrum whose smallest eigenvalues are genuine low-curvature bending directions
that nonetheless carry large variance over the data. Under a single REML ``λ``
those near-null modes cost almost nothing yet absorb EDF freely, leaving a flat
REML profile whose over-smoothed ``λ → ∞`` corner is a spurious optimum the outer
optimizer slides into on certain data clouds (shrinking every smooth coordinate
AND the affine null space to EDF = 1). The thin-plate path already cured the same
pathology in #1347 by reparameterizing in the realized DATA METRIC — solving the
generalized eigenproblem ``Ω_c v = μ G_c v`` with ``G_c = (K Z)ᵀ(K Z)`` so the
spectrum becomes curvature-per-unit-data-variance (mgcv's sharp cliff). The fix
routes the Duchon kernel block through that SAME data-metric reparam: it folds
the ``G_c``-orthonormal generalized eigenbasis ``V`` into the constrained kernel
transform (``Z' = Z·V``) so the realized design ``K·Z·V`` rotates into that basis
and the native penalty becomes the diagonal ``diag(μ)``, and freezes ``V`` so
predict / κ-trial / ψ-derivative rebuilds replay the exact fit-time rotation.

The matern collapse in #1357 is a distinct mechanism (the isotropic-κ outer
optimization axis) and is intentionally not covered here.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


# The exact seeds that collapsed on `origin/main` before the fix (4/20 over
# range(20); these four are the canonical regression set from the issue).
COLLAPSE_SEEDS = (0, 2, 4, 7)


def _truth(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    # Smooth, clearly non-polynomial surface so a degenerate plane/constant fit
    # cannot accidentally clear the R² bar.
    return np.sin(2.0 * x1) + np.cos(1.5 * x2) + 0.5 * x1 * x2


def _fit_quality(seed: int, n: int = 300) -> tuple[float, float]:
    """Return (in-sample R² against the noise-free truth, prediction std)."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = rng.uniform(-1.0, 1.0, n)
    truth = _truth(x1, x2)
    y = truth + rng.normal(0.0, 0.1, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    model = gamfit.fit(df, "y ~ duchon(x1,x2)")
    pred = np.asarray(model.predict(df[["x1", "x2"]])).reshape(-1)
    ss_res = float(np.sum((pred - truth) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return r2, float(pred.std())


@pytest.mark.parametrize("seed", COLLAPSE_SEEDS)
def test_duchon_default_does_not_collapse_on_known_collapse_seeds(seed: int) -> None:
    """Each formerly-collapsing seed recovers the signal with a non-degenerate
    surface. The threshold (0.8) sits far above a collapsed constant fit (R²≈0)
    and comfortably below a correct fit (R²≈0.998)."""
    r2, pred_std = _fit_quality(seed)
    assert r2 > 0.8, f"duchon(x1,x2) seed={seed} collapsed: R²={r2:.4f} (expected > 0.8)"
    # A collapsed fit has prediction std ~1e-4; the signal range is ~order 1.
    assert pred_std > 0.1, (
        f"duchon(x1,x2) seed={seed} produced a near-flat surface: std={pred_std:.2e}"
    )


def test_duchon_default_recovers_across_many_draws() -> None:
    """Aggregate guard over 20 draws: the median fit is excellent and NO draw
    collapses. Pre-fix this set had 4/20 collapses (R²≈0)."""
    r2s = [_fit_quality(seed)[0] for seed in range(20)]
    median_r2 = float(np.median(r2s))
    n_collapsed = sum(1 for r in r2s if r < 0.8)
    assert median_r2 > 0.95, f"median duchon R²={median_r2:.4f} over 20 draws (expected > 0.95)"
    assert n_collapsed == 0, (
        f"{n_collapsed}/20 duchon fits collapsed (R² < 0.8); "
        f"R² values={[round(r, 3) for r in r2s]}"
    )


def test_duchon_default_rotation_does_not_trigger_collapse() -> None:
    """Rotation is just another data realization, and the pre-fix bug exposed
    itself as a loss of rotation invariance: with the DEFAULT centers, one
    orientation fit (EDF≈36) while a rigidly rotated copy collapsed (EDF≈1),
    giving predictions that differed by ~45% of the signal range.

    After the fix both orientations fit, so corresponding-point predictions agree
    to a small fraction of the signal range (the residual difference is the
    benign farthest-point center reshuffle, NOT a collapse). We assert well below
    the ~0.45 collapse signature.
    """
    phi = 0.7
    c, s = np.cos(phi), np.sin(phi)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        n = 300
        x1 = rng.uniform(-1.0, 1.0, n)
        x2 = rng.uniform(-1.0, 1.0, n)
        truth = _truth(x1, x2)
        y = truth + rng.normal(0.0, 0.1, n)
        r1 = c * x1 - s * x2
        r2v = s * x1 + c * x2

        d0 = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        d1 = pd.DataFrame({"x1": r1, "x2": r2v, "y": y})
        m0 = gamfit.fit(d0, "y ~ duchon(x1,x2)")
        m1 = gamfit.fit(d1, "y ~ duchon(x1,x2)")
        p0 = np.asarray(m0.predict(d0[["x1", "x2"]])).reshape(-1)
        p1 = np.asarray(m1.predict(d1[["x1", "x2"]])).reshape(-1)

        sig_range = float(truth.max() - truth.min())
        rel_diff = float(np.max(np.abs(p0 - p1))) / sig_range
        assert rel_diff < 0.2, (
            f"seed={seed}: rotated duchon fit differs by {rel_diff:.3f} of the "
            f"signal range (collapse signature is ~0.45)"
        )
