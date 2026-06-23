"""Bug hunt: ``model.sample()`` ignores ``nonnegative()`` / ``linear(min=, max=)``
box constraints on a linear coefficient — the posterior draws routinely land on
the *forbidden* side of an active bound.

A ``nonnegative(x)`` term constrains its coefficient to ``β ≥ 0``; ``linear(x,
min=a, max=b)`` constrains ``a ≤ β ≤ b``. These are enforced at fit time as KKT
inequality rows (``src/terms/smooth/design_construction.rs:160-174`` reads
``LinearTermSpec.coefficient_min`` / ``coefficient_max``), so the *point
estimate* correctly pins to the boundary when the constraint is active. The
posterior must respect the same admissible region: a coefficient that is not
allowed to be negative cannot have half its posterior mass below zero.

It does. When the true effect points away from the feasible region (so the bound
is active and the MLE sits exactly on it), the posterior draws are a plain,
unconstrained Gaussian ``N(mode, φ·H⁻¹)`` centred on the boundary — putting
roughly half the mass strictly outside the constraint. The cause is that the
posterior sampler never reads the box bounds: ``coefficient_min`` /
``coefficient_max`` appear in the parser
(``src/inference/formula_dsl.rs:2463-2491``) and the fit-time KKT assembly, but
**not anywhere in** ``src/inference/sample.rs``. ``sample_standard`` /
``laplace_gaussian_fallback`` draw an unconstrained Gaussian on the user scale
and report it with ``is_exact`` / ``rhat ≈ 1`` — a confidently wrong posterior.

This is distinct from the ``bounded(...)`` interval term, which *does* have a
dedicated latent-logit sampler (``sample_standard_bounded``); the box
constraints ``nonnegative`` / ``nonpositive`` / ``linear(min,max)`` /
``constrain`` have no posterior handling at all, and the violation reproduces for
both Gaussian and non-Gaussian families.

This test fits a Gaussian model whose unconstrained slope would be strongly
negative under a ``nonnegative(x)`` constraint (so the bound is active and the
fitted slope is exactly 0), draws the posterior, and asserts that essentially all
draws of the constrained coefficient are ``≥ 0`` (a tiny tolerance absorbs
round-off). It currently fails — about half the draws are negative. When the
sampler respects the constraint, the assertion holds without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _slope_draws(model: Any, frame: pd.DataFrame) -> np.ndarray:
    samples = model.sample(frame, samples=2000, chains=2, seed=1)
    names = list(samples.coefficient_names)
    # The single non-intercept coefficient is the constrained slope.
    slope_idx = names.index("beta_1") if "beta_1" in names else len(names) - 1
    return np.asarray(samples.to_numpy())[:, slope_idx]


def test_posterior_respects_nonnegative_coefficient_bound() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal(n)
    # True slope is strongly NEGATIVE, so a `nonnegative(x)` constraint is active
    # and the fitted slope pins to exactly 0.
    y = -3.0 * x + rng.standard_normal(n) * 0.5
    frame = pd.DataFrame({"x": x, "y": y})

    model = gamfit.fit(frame, "y ~ nonnegative(x)")
    fitted_slope = model.summary().coefficients[1]["estimate"]
    # Sanity: the constraint is active (fitted slope on the boundary).
    assert abs(fitted_slope) < 1e-6, f"expected an active bound (slope≈0); got {fitted_slope}"

    draws = _slope_draws(model, frame)
    frac_negative = float((draws < -1e-8).mean())
    assert frac_negative < 0.01, (
        "posterior of a nonnegative()-constrained coefficient must stay in [0, ∞); "
        f"got {frac_negative:.1%} of draws below 0 (min draw = {draws.min():.4f})"
    )


def test_posterior_respects_two_sided_linear_coefficient_bounds() -> None:
    rng = np.random.default_rng(2)
    n = 200
    x = rng.standard_normal(n)
    # True slope -3 is below the lower bound -1, so the box constraint is active
    # and the fitted slope pins to exactly -1.
    y = -3.0 * x + rng.standard_normal(n) * 0.5
    frame = pd.DataFrame({"x": x, "y": y})

    model = gamfit.fit(frame, "y ~ linear(x, min=-1, max=1)")
    fitted_slope = model.summary().coefficients[1]["estimate"]
    assert abs(fitted_slope - (-1.0)) < 1e-6, (
        f"expected an active lower bound (slope≈-1); got {fitted_slope}"
    )

    draws = _slope_draws(model, frame)
    frac_outside = float(((draws < -1.0 - 1e-8) | (draws > 1.0 + 1e-8)).mean())
    assert frac_outside < 0.01, (
        "posterior of a linear(min=-1, max=1)-constrained coefficient must stay in "
        f"[-1, 1]; got {frac_outside:.1%} of draws outside (range "
        f"[{draws.min():.4f}, {draws.max():.4f}])"
    )
