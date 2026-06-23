"""Bug hunt #1514: ``bounded(x, min, max)`` coefficient standard errors are
~20× too wide on a Gaussian model.

A ``bounded(x, min, max)`` coefficient that is interior and well-identified
recovers the SAME point estimate as the unconstrained linear term, but the
fit exported its coefficient covariance as the raw inverse penalized Hessian
``H_user⁻¹`` and hardcoded ``Dispersion::Known(1.0)``. For a profiled-Gaussian
family the IRLS working weight is scale-free (``W = priorweights``), so
``H_user`` carries unit implicit dispersion and the reported ``Vb`` must be
restored to ``σ̂²·H_user⁻¹`` with the REML residual variance
``σ̂² = RSS/(n − edf)``. Omitting that scale made the bounded SE
``≈ 1/√Σ(xᵢ−x̄)²`` instead of ``σ̂/√Σ(xᵢ−x̄)²`` — i.e. ``~1/σ̂`` (≈20× at
σ̂ ≈ 0.05) too wide.

Angles covered here:

* **Width parity with the unconstrained control** — the reported failure: a
  bounded Gaussian slope's ``std_error`` must match the equivalent
  unconstrained slope's ``std_error`` (same data, same OLS information).
* **Sampler/fit consistency** — the posterior draw std must track the reported
  ``std_error``; the latent sampler re-applies ``√σ̂²`` so support AND width are
  both right (distinct from #1508, which fixed only support).
* **Dispersion scaling law** — a second, independent angle on the same root
  cause: the bounded SE must scale *linearly* with the residual noise σ. The
  buggy export was σ-independent (it dropped the σ̂² factor), so doubling the
  noise left the SE unchanged; the fixed export doubles it.
* **Binomial control** — fixed-scale (φ ≡ 1) bounded fits must be unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def _se(model) -> float:
    return float(model.summary().coefficients[1]["std_error"])


def _est(model) -> float:
    return float(model.summary().coefficients[1]["estimate"])


def test_bounded_gaussian_se_matches_unconstrained() -> None:
    """The interior bounded slope SE must equal the unconstrained slope SE."""
    rng = np.random.default_rng(20)
    n = 400
    x = rng.uniform(0, 1, n)
    y = 1.0 + 0.5 * x + rng.standard_normal(n) * 0.05
    df = pd.DataFrame({"x": x, "y": y})

    mb = gamfit.fit(df, "y ~ bounded(x, min=0, max=1)")
    mu = gamfit.fit(df, "y ~ x")

    # Same point estimate (interior, well away from either bound).
    assert abs(_est(mb) - _est(mu)) < 1e-3, (_est(mb), _est(mu))

    se_bounded, se_unconstr = _se(mb), _se(mu)
    # Before the fix this ratio was ~20 (= 1/σ̂). It must now be ~1.
    ratio = se_bounded / se_unconstr
    assert 0.9 < ratio < 1.1, (
        f"bounded SE {se_bounded:.5f} vs unconstrained {se_unconstr:.5f} "
        f"(ratio {ratio:.2f}; was ~20× too wide)"
    )


def test_bounded_gaussian_draw_std_tracks_summary_se() -> None:
    """The latent sampler width must match the corrected exported SE."""
    rng = np.random.default_rng(20)
    n = 400
    x = rng.uniform(0, 1, n)
    y = 1.0 + 0.5 * x + rng.standard_normal(n) * 0.05
    df = pd.DataFrame({"x": x, "y": y})

    mb = gamfit.fit(df, "y ~ bounded(x, min=0, max=1)")
    se = _se(mb)
    draws = mb.sample(df, samples=8000, chains=2, seed=1).to_numpy()[:, 1]
    draw_std = float(draws.std())

    # Fit export and sampler must agree (the sampler re-applies √σ̂²).
    assert abs(draw_std - se) / se < 0.15, (
        f"posterior draw std {draw_std:.5f} disagrees with summary SE {se:.5f}"
    )
    # And both must be order σ̂/√Σ(x−x̄)² ≈ 0.0085, not ~0.17.
    assert se < 0.03, f"bounded SE {se:.5f} still inflated"


def test_bounded_gaussian_se_scales_with_noise() -> None:
    """Independent angle: SE must scale linearly with the residual noise σ.

    The buggy export dropped the σ̂² factor, making the SE σ-independent. The
    correct SE ∝ σ̂, so a 4× noise increase must give ~4× the SE.
    """
    rng = np.random.default_rng(7)
    n = 500
    x = rng.uniform(0, 1, n)
    base = 1.0 + 0.5 * x

    def fit_se(sigma: float) -> float:
        # Reuse one noise vector so only its amplitude changes.
        noise = rng.standard_normal(n)
        df = pd.DataFrame({"x": x, "y": base + noise * sigma})
        return _se(gamfit.fit(df, "y ~ bounded(x, min=0, max=1)"))

    se_lo = fit_se(0.02)
    se_hi = fit_se(0.08)
    ratio = se_hi / se_lo
    # Expect ~4× (linear in σ); the buggy export gave ~1× (no scaling).
    assert 3.0 < ratio < 5.0, (
        f"SE ratio {ratio:.2f} for a 4× noise increase — SE is not scaling with σ"
    )


def test_binomial_bounded_se_unchanged_control() -> None:
    """Fixed-scale (φ ≡ 1) bounded fits must match the unconstrained control.

    The bound is wide enough (``[-3, 3]``) that the true slope stays well
    interior — at a boundary the interval-transform Jacobian collapses and the
    user-scale SE legitimately shrinks toward zero, which would confound the
    parity check.
    """
    rng = np.random.default_rng(7)
    n = 800
    x = rng.uniform(0, 1, n)
    p = 1.0 / (1.0 + np.exp(-(-0.5 + 0.7 * x)))
    y = (rng.uniform(size=n) < p).astype(float)
    df = pd.DataFrame({"x": x, "y": y})

    mb = gamfit.fit(df, "y ~ bounded(x, min=-3, max=3)", family="binomial")
    mu = gamfit.fit(df, "y ~ x", family="binomial")

    # Confirm the bounded slope is interior so the Jacobian is healthy.
    assert -3.0 < _est(mb) < 3.0
    se_bounded, se_unconstr = _se(mb), _se(mu)
    ratio = se_bounded / se_unconstr
    assert 0.95 < ratio < 1.05, (
        f"binomial bounded SE {se_bounded:.5f} vs unconstrained {se_unconstr:.5f}"
    )
