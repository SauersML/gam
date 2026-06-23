"""Bug hunt: posterior draws from a ``shape='monotone_increasing'`` smooth are
not monotone — the sampler draws an unconstrained Gaussian on the spline
coefficients and most drawn curves go *down*.

A shape-constrained smooth ``s(x, shape='monotone_increasing')`` restricts the
spline coefficients to the monotone cone: the higher-order coefficients are
bounded below by 0 and enforced at fit time as linear inequality rows
(``src/terms/smooth/shape_constraints.rs`` — ``shape_lower_bounds_local`` sets
``lb[j]=0`` for ``j >= order``, assembled into ``LinearInequalityConstraints``).
The fitted point estimate therefore is monotone. The parameter space of the
model *is* the monotone cone, so the posterior over curves must live there too:
every posterior draw must be a monotone-increasing curve.

It does not. ``model.sample()`` draws a plain Gaussian ``N(mode, φ·H⁻¹)`` on the
raw spline coefficients with no awareness of the shape inequalities —
``src/inference/sample.rs`` contains no reference to ``monotone`` / shape
constraints at all — so the drawn curves routinely decrease. This is the
smooth-term sibling of the linear-coefficient box-constraint sampling defect
(see Related issues): the constraint is honored by the fit and by point
prediction but silently dropped by the posterior sampler, which still reports
``is_exact`` / ``rhat ≈ 1``.

This test fits a clean monotone-increasing signal, confirms the *fitted* curve
is monotone (the constraint works at fit time), then draws the posterior
predictive curves on a dense grid and asserts that essentially all of them are
monotone increasing (a tolerance of 1% of the signal range absorbs round-off).
It currently fails — about three quarters of the drawn curves contain a clear
decrease. When the sampler respects the monotone cone, the assertion holds
without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def test_posterior_draws_of_monotone_smooth_are_monotone() -> None:
    rng = np.random.default_rng(0)
    n = 300
    x = np.sort(rng.uniform(0.0, 1.0, n))
    # Strongly monotone-increasing signal.
    y = np.log(x + 0.1) + rng.standard_normal(n) * 0.1
    frame = pd.DataFrame({"x": x, "y": y})

    model = gamfit.fit(frame, "y ~ s(x, shape='monotone_increasing')")

    grid = pd.DataFrame({"x": np.linspace(0.0, 1.0, 80)})
    # Sanity: the fitted curve honors the constraint.
    fitted = np.asarray(model.predict(grid)).ravel()
    assert np.diff(fitted).min() >= -1e-6, (
        f"fitted monotone curve must be increasing; got min step {np.diff(fitted).min()}"
    )

    samples = model.sample(frame, samples=600, chains=2, seed=1)
    predictive = samples.predict_draws(grid)
    curves = np.asarray(predictive.mean)  # (n_draws, n_grid)
    if curves.shape[1] != len(grid):
        curves = curves.T
    assert curves.shape[1] == len(grid)

    signal_range = float(np.ptp(curves))
    # A real violation is a decrease that is a non-trivial fraction of the curve's
    # own scale, not float round-off (the fitted curve's smallest step is ~0.07%
    # of the range, so 0.5% is comfortably above the noise floor).
    tol = 0.005 * max(signal_range, 1.0)
    worst_drop_per_draw = np.array([np.diff(c).min() for c in curves])
    frac_non_monotone = float((worst_drop_per_draw < -tol).mean())

    assert frac_non_monotone < 0.05, (
        "posterior draws of a monotone_increasing smooth must stay in the monotone "
        f"cone; {frac_non_monotone:.1%} of {len(curves)} drawn curves decrease by more "
        f"than 0.5% of the signal range (worst drop {worst_drop_per_draw.min():.4f} on a "
        f"range of {signal_range:.3f})"
    )
