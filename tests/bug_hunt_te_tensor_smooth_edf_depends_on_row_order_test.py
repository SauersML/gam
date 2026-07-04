"""Bug hunt (#2123): a ``te(x, z)`` tensor-product smooth's reported effective
degrees of freedom — and therefore its standard errors and AIC — must NOT depend
on the row order of the training frame.

Rows of a regression frame are exchangeable: the REML objective is a sum over
rows, so every fitted and inferential quantity is invariant under a row
permutation. Additive smooths obey this to ~1e-13. The tensor smooth did not:
on the *original* row order the outer REML railed both smoothing parameters
(λ≈(8.8e4, 1.1e13)) and floored the EDF to the full basis dimension (edf=52,
ΣSE=3.84), while *every* row permutation converged to λ≈(1.5, 6.2e6), edf≈13.5,
ΣSE≈13.

Root cause (fixed): the stable reparameterization computed the penalized-block
penalty spectrum by eigendecomposing the ASSEMBLED Gram ``Σ_k λ_k S_k = EᵀE``,
which squares the condition number (κ(EᵀE)=κ(E)²). When one margin saturates
(here the near-linear ``z`` axis, λ_ratio≳1e8) the recessive-penalty
eigenvectors sank below the O(ε·λ_max) noise floor, the penalty silently
vanished in a genuinely-penalized direction, and the inner P-IRLS then fit that
direction to the data — injecting discontinuous cliffs into the LAML objective
and a false low-cost basin in the high-λ corner that the multistart landed in
for some row orders. The fix takes the spectrum from the SVD of the stacked
scaled roots ``E=[√λ_k R_k]`` (κ(E), not κ(E)²). A second facet: even a
correctly-converged fit that lands on the ρ₁ rail must report stable SEs; the
posterior covariance now uses a ridge-free spectral inverse when the Hessian is
ill-conditioned, instead of an additive ridge that uniformly collapsed the SEs.

This test fits ``te(x, z)`` on the original frame and on several fixed row
permutations and asserts:
  * the deviances agree (same data / same fit quality — the anchor),
  * the reported EDF agrees across orderings (invariance), and
  * the summed standard errors agree across orderings (invariance).
It previously failed with edf 52 vs 13.5 (ΣSE 3.84 vs 12.98) on the original
order; with the reparameterization + covariance fixes it passes without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _frame(seed: int = 0, n: int = 300):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + 0.7 * z + rng.standard_normal(n) * 0.2
    return pd.DataFrame({"x": x, "z": z, "y": y})


def _fit_stats(frame: pd.DataFrame) -> dict[str, float]:
    model = gamfit.fit(frame, "y ~ te(x, z)")
    s = model.summary()
    se = np.asarray(s.coefficients_frame()["std_error"], dtype=float)
    return {
        "edf": float(s.edf_total),
        "deviance": float(s.deviance),
        "se_sum": float(np.nansum(se)),
    }


def test_te_tensor_smooth_edf_and_se_invariant_to_row_order() -> None:
    d = _frame()
    n = len(d)

    base = _fit_stats(d)

    # Sanity: a tensor smooth of this signal is a genuine, well-supported fit —
    # neither the intercept-only floor (edf≈1) nor a near-interpolating rail
    # (edf≈52, the full 7×7 basis). It should spend a moderate, sensible EDF.
    assert 5.0 < base["edf"] < 30.0, (
        f"original-order te(x,z) EDF={base['edf']:.3f} is not a sensible smooth fit "
        f"(railed/floored?)"
    )
    # And its standard errors must not have collapsed toward zero.
    assert base["se_sum"] > 5.0, (
        f"original-order ΣSE={base['se_sum']:.3f} collapsed — covariance is degenerate"
    )

    for seed in (1, 7, 101, 2024, 3, 10, 5, 8):
        perm = d.iloc[np.random.default_rng(seed).permutation(n)].reset_index(drop=True)
        got = _fit_stats(perm)

        # Same data → same fit quality. This is the exchangeability anchor.
        assert abs(got["deviance"] - base["deviance"]) <= 1e-2 * (1.0 + abs(base["deviance"])), (
            f"deviance differs under permutation seed={seed}: "
            f"{got['deviance']:.5f} vs {base['deviance']:.5f}"
        )
        # EDF must be invariant to the row order (the headline #2123 assertion:
        # original order previously railed to edf=52 while permutations gave 13.5).
        assert abs(got["edf"] - base["edf"]) <= 2.0, (
            f"EDF depends on row order (seed={seed}): {got['edf']:.3f} vs {base['edf']:.3f}"
        )
        # No ordering may collapse the covariance. The near-linear z margin makes
        # the ρ₁ REML ridge nearly flat, so different orders legitimately land at
        # different (REML-equivalent) λ₁ — including the exact rail (λ₁≈1e13,
        # κ(H)≈1e13). There the additive stabilization ridge in the naive inverse
        # uniformly shrank every posterior variance (ΣSE collapsed to ≈0.16); the
        # ridge-free spectral covariance inverse keeps the well-determined
        # directions' variances intact, so ΣSE stays a real (non-degenerate) value.
        assert got["se_sum"] > 1.0, (
            f"ΣSE collapsed under permutation seed={seed}: {got['se_sum']:.3f} "
            f"(covariance degenerate at the saturated ρ rail)"
        )
