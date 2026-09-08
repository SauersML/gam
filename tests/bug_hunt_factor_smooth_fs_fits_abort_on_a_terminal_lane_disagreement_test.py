"""Bug hunt: ``y ~ s(x) + s(x, g, bs='fs')`` aborts on **half** of ordinary
Gaussian datasets, because the terminal certificate finds the outer criterion's
two evaluation lanes disagreeing at the same rho by up to 2180x their own
roundoff bound.

Sweeping ``seed in range(40)`` over the fixture below (n=120, 3 groups,
``y = sin(6x) + 0.3 N(0,1)``, no group effect):

* **20 of 40 fits raise** ``IntegrationError`` -- all twenty with the same
  message,

      Outer smoothing-parameter optimization did not certify a stationary
      optimum (standard REML): cost-only value disagrees with analytic-sample
      value at the same outer point: value-only=4.9108012271264322e1,
      analytic-sample=4.9107986388870351e1, disagreement=2.588e-5,
      roundoff bound=7.318e-7, inner solve: value-lane=converged,
      derivative-lane=converged

* the other 20 fit fine and recover the signal: ``corr(predicted, sin(6x))``
  has minimum ``0.9893`` and median ``0.9946``.  The model class is healthy;
  what fails is the audit.

Across the twenty failures the disagreement runs from ``2.588e-5`` to
``9.971e-4`` while the bound runs from ``4.028e-7`` to ``7.636e-7`` -- a
**35x to 2180x** overshoot, not a borderline tolerance.  Every failure is
bit-reproducible: refitting the same seed reproduces the identical
``value-only`` / ``analytic-sample`` pair, so this is not a stochastic trace
estimator.

What the check is (``audit_outer_value_agreement``,
``crates/gam-solve/src/rho_optimizer/run.rs:3868``): at terminal certification
the same criterion is priced twice at the same rho -- once through
``OuterEvalOrder::Value`` and once through the derivative-bearing lane -- and
the two must agree within ``outer_value_agreement_bound`` (``run.rs:3860``),
``sqrt(eps) * max(|v1|, |v2|, 1)``.  The bound's stated premise is that the two
lanes "may use different kernels and reduction trees, so bitwise identity is not
a valid contract"; it is a *roundoff* envelope.  A deterministic 2180x overshoot
with both lanes reporting ``inner solve: converged`` is not roundoff between two
reduction trees -- the lanes are pricing different quantities.

Where they diverge (``compute_outer_eval_with_order``,
``crates/gam-solve/src/reml/objective.rs:2780``): the two orders take
structurally different routes.  ``OuterEvalOrder::Value`` short-circuits at
line 2842 into ``evaluate_unified(..., EvalMode::ValueOnly)`` and is
**deliberately not cached** ("Deliberately NOT cached into ``store_outer_eval``",
line 2873); the derivative-bearing orders fall through to the
``EvalMode::ValueAndGradient`` / ``ValueGradientHessian`` assembly and *are*
served from / written to the LRU keyed by ``rhokey_sanitized``
(``crates/gam-solve/src/reml/gradient_hessian.rs:4133``), whose identity also
carries ``screening_max_inner_iterations`` and ``outer_inner_cap``.  A
multi-lambda ``fs`` block -- one lambda per group plus the shared smooth -- is
exactly the configuration that makes the two routes' inner states drift apart.

Where the search is when it happens: every failure reports an
``rho_checkpoint`` with at least one coordinate pinned near the ``RHO_BOUND``
ceiling of ``30`` (``crates/gam-solve/src/estimate/smoothing_correction.rs:110``),
i.e. ``lambda ~ 1e12-1e13``.  Across the twenty failures ``max(rho)`` runs
``[27.22, 30.00]``, is above ``28`` in 16 of 20 and exactly ``30.0`` in 7 of 20;
across the twenty successes ``max(log lambda)`` runs ``[15.50, 25.89]`` and
exceeds ``25`` only once.  That is a clean separation, and it is the same
float64 wall that #2830 and #2831 hit from the other side -- there the
natural-parameterization normal matrix at ``lambda ~ 1e13`` fails its Cholesky
outright; here two lanes that assemble it differently simply stop agreeing.
The rail is not gratuitous either: ``y ~ s(x) + s(x, g, bs='fs')`` on data with
no group effect *should* send the per-group deviation lambdas to infinity.

Consistent with that, the failure needs the shared smooth AND the ``fs`` block:
over the same 24 seeds, ``y ~ s(x, g, bs='fs')`` alone, ``bs='sz'``, ``bs='re'``,
``group(g)`` and plain ``s(x)`` all fit 24/24, ``s(x) + s(x, by=g)`` fails 2/24
with the identical message, and ``s(x) + s(x, g, bs='fs')`` fails 13/24.  The
rate falls as the rail gets harder to reach: 10 groups -> 4/24, ``n=600`` ->
3/16.

This is a *user-facing* abort of a first-class documented term type
(``bs='fs'`` in ``docs/formulas.md``), on a data-generating process with no
pathology in it: 120 rows, 3 balanced groups, one smooth signal, Gaussian noise.

The assertions are fix-agnostic: the model must fit and must recover the signal
it is given. Nothing about lanes, caches, or tolerances is named -- reconciling
the lanes, or widening the audit only where it is genuinely roundoff, both make
this pass.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

import gamfit

_N = 120
_N_GROUPS = 3
# Seeds observed to abort at HEAD, plus two that already fit, as controls.
_FAILING_SEEDS = [0, 2, 3, 6, 7, 10, 11, 12, 16, 17]
_PASSING_SEEDS = [1, 4]


def _dataset(seed: int) -> tuple[dict[str, npt.NDArray[np.generic]], npt.NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, _N))
    g = rng.integers(0, _N_GROUPS, _N).astype(str)
    mu = np.sin(6.0 * x)
    y = mu + 0.3 * rng.normal(size=_N)
    return {"x": x, "g": g, "y": y}, mu


@pytest.mark.parametrize("seed", _FAILING_SEEDS + _PASSING_SEEDS)
def test_factor_smooth_fs_fits_and_recovers_its_signal(seed: int) -> None:
    data, mu = _dataset(seed)

    model = gamfit.fit(data, "y ~ s(x) + s(x, g, bs='fs')")

    predicted = np.asarray(
        model.predict(data, return_type="pandas")["posterior_mean"], dtype=np.float64
    )
    assert np.isfinite(predicted).all()
    # The twenty seeds that already fit at HEAD reach corr >= 0.9893
    # (median 0.9946), so 0.95 is a floor with real slack that still rejects a
    # "fix" that returns a flat or garbage fit.
    correlation = float(np.corrcoef(predicted, mu)[0, 1])
    assert correlation > 0.95, (
        f"fs fit recovered the signal at corr={correlation:.4f}"
    )
