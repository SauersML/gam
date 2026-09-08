"""Bug hunt: ``gamfit.gaussian_reml_fit_blocks_forward`` turns a *point-local*
numerical breakdown at one trial ``lambda`` into a fatal, whole-fit abort.

The multi-block Gaussian-REML driver (documented at ``docs/api-reference.md:348``
and exported as ``gamfit.gaussian_reml_fit_blocks_forward``) profiles the REML
score over ``rho = log(lambda)`` with an outer ARC search.  Every trial ``rho``
the search visits is evaluated by ``GaussianRemlBlocksProfile::evaluate``
(``crates/gam-solve/src/gaussian_reml.rs:385``), which assembles the penalized
normal matrix ``X'WX + sum_k lambda_k S_k`` **in the natural parameterization**
and demands a strict SPD factorization of it
(``certified_spd_inverse`` at line ~397, a second ``cholesky`` at line ~415, and
the ``q > 0`` profiled-residual guard at line ~486).

Those three guards are all ``rho``-local: for a fixed, perfectly well-conditioned
design they hold at moderate ``lambda`` and break down in float64 once a single
``lambda_k`` reaches ~1e12-1e13, because the assembled matrix's condition number
grows like ``lambda_k`` and its smallest eigenvalue goes negative in rounding.
That is exactly the regime the search *walks into on its own* whenever one
block's REML optimum sits at ``lambda -> inf`` (the everyday "this smooth is
already inside its own penalty null space" case).

But the errors are raised as ``EstimationError::InvalidInput``, and
``EstimationError::is_trial_point_infeasible``
(``crates/gam-problem/src/estimation_error.rs:935``) classifies ``InvalidInput``
as **not** a trial-point refusal.  ``into_objective_error``
(``crates/gam-solve/src/rho_optimizer/objective.rs:1545``) therefore stamps it
``ObjectiveEvalKind::Fatal``, and the ARC arm at
``crates/gam-solve/src/rho_optimizer/run_plan.rs:2810`` converts it into
``EstimationError::fatal_objective_evaluation("outer ARC evaluation", ...)`` --
killing the entire fit.  The machinery for the correct behaviour already exists
one line up in the same table: ``EstimationError::TrialPointRefused`` (line 898),
which the bridge comment at ``crates/gam-solve/src/rho_optimizer/bridges.rs:3929``
describes as the whole point -- "a REML/inner-solve failure is POINT-LOCAL
evidence about this rho ... blanket-classifying it fatal turned one invalid
startup seed into a whole-fit 'Fatal outer-objective evaluation failure'".
The block-REML profile never got that treatment.

Observed, on the fixed dataset below (n=60, two cubic B-spline blocks, one
sum-to-zero constrained, strictly positive weights, and a joint coefficient map
the driver's own rank certificate accepts):

* the unperturbed fit succeeds, reporting ``lambda = [6.3e-04, 2.0e+05]``;
* of the 120 single-coordinate ``+/- 1e-6`` perturbations of ``y``, **11 raise**
  ``InvalidInputError`` -- either
  ``"... requires an exact SPD penalized normal matrix: Cholesky factorization
  failed: NonPositivePivot { index: 14 }"`` or
  ``"... profiled residual quadratic form must be finite and positive; got
  -0.3138525085294064"``;
* the 109 that *do* succeed move the fitted values by at most ``3.9e-07`` and
  the REML score by at most ``1.6e-05``.  The fit is perfectly continuous in
  ``y``; the failures are not a real discontinuity, they are an aborted search.
* the same abort is reachable directly: with the response held fixed and only
  the *starting point* changed, ``init_rhos = [-7.37, 28.0]``
  (``lambda_2 = e^28 ~ 1.4e12``) raises, while ``25.0``, ``30.0``, ``32.0``,
  ``35.0`` and ``40.0`` all return the identical optimum.  A starting point
  cannot make a feasible problem infeasible.

Expected: an SPD/positivity breakdown at a trial ``rho`` is evidence about that
``rho`` only.  It must be reported as an infeasible trial point (``+inf`` cost /
``TrialPointRefused``) so the trust region rejects the step and continues, exactly
as ``PirlsDidNotConverge`` and ``ModelIsIllConditioned`` already are.  The fit
that the very same objective certifies at neighbouring starting points must be
returned.

The assertions below are fix-agnostic: they never name ``lambda`` or a solver
path, only that (a) the call returns for every starting point in a grid, and
(b) a ``1e-6`` nudge of a single response value moves the fitted vector by less
than ``1e-4``.  Both hold for any repair -- reclassifying the refusal, adding a
stable reparameterization, or bounding ``rho``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import gamfit

_K_INTERNAL_KNOTS = 4
_DEGREE = 3
_PENALTY_ORDER = 2
_N = 60


def _spline_block(
    t: npt.NDArray[np.float64], *, constrain: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A cubic B-spline design + exact second-derivative roughness penalty.

    ``constrain=True`` applies the usual sum-to-zero side condition so that the
    two blocks do not both carry the constant (which the driver's own rank
    certificate would -- correctly -- reject as unidentified).
    """
    basis = np.asarray(
        gamfit.bspline_basis(t, _K_INTERNAL_KNOTS, degree=_DEGREE), dtype=np.float64
    )
    interior = np.quantile(t, np.linspace(0.0, 1.0, _K_INTERNAL_KNOTS + 2)[1:-1])
    knots = np.concatenate(
        [
            np.repeat(t.min(), _DEGREE + 1),
            interior,
            np.repeat(t.max(), _DEGREE + 1),
        ]
    )
    penalty = np.asarray(
        gamfit.smoothness_penalty(knots, degree=_DEGREE, order=_PENALTY_ORDER)[0],
        dtype=np.float64,
    )
    if not constrain:
        return basis, penalty
    column_sums = basis.sum(axis=0).reshape(-1, 1)
    q_full, _ = np.linalg.qr(column_sums, mode="complete")
    z = q_full[:, 1:]
    return basis @ z, z.T @ penalty @ z


def _problem() -> tuple[
    list[npt.NDArray[np.float64]],
    list[npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    rng = np.random.default_rng(5)
    x1 = np.sort(rng.uniform(0.0, 1.0, _N))
    x2 = rng.uniform(0.0, 1.0, _N)
    design_1, penalty_1 = _spline_block(x1, constrain=False)
    design_2, penalty_2 = _spline_block(x2, constrain=True)
    y = np.sin(4.0 * x1) + 0.5 * x2**2 + 0.15 * rng.normal(size=_N)
    weights = rng.uniform(0.5, 2.0, _N)
    return [design_1, design_2], [penalty_1, penalty_2], y, weights


def _fit(
    designs: list[npt.NDArray[np.float64]],
    penalties: list[npt.NDArray[np.float64]],
    y: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    init_rhos: npt.NDArray[np.float64] | None = None,
) -> dict[str, object]:
    return gamfit.gaussian_reml_fit_blocks_forward(
        designs, penalties, y, weights=weights, init_rhos=init_rhos
    )


def test_block_reml_forward_is_continuous_under_a_1e_minus_6_response_nudge() -> None:
    """A 1e-6 nudge of one response value must not abort the fit.

    The engine's own successful neighbours move ``fitted`` by <= 3.9e-07, so a
    1e-4 ceiling is three orders of magnitude of slack; the point of the test is
    that the call must *return* at all.
    """
    designs, penalties, y, weights = _problem()

    base = _fit(designs, penalties, y, weights)
    base_fitted = np.asarray(base["fitted"], dtype=np.float64).ravel()
    base_score = float(np.asarray(base["reml_score"], dtype=np.float64))
    assert np.isfinite(base_fitted).all()
    assert np.isfinite(base_score)

    # Indices observed to abort at HEAD, plus their signs.
    nudges = [(4, -1.0), (11, -1.0), (13, 1.0), (19, 1.0), (25, -1.0), (27, 1.0)]
    for index, sign in nudges:
        perturbed = y.copy()
        perturbed[index] += sign * 1e-6
        fit = _fit(designs, penalties, perturbed, weights)
        fitted = np.asarray(fit["fitted"], dtype=np.float64).ravel()
        score = float(np.asarray(fit["reml_score"], dtype=np.float64))
        assert np.abs(fitted - base_fitted).max() < 1e-4, (
            f"y[{index}] {sign:+.0e} moved the fitted vector by "
            f"{np.abs(fitted - base_fitted).max():.3e}"
        )
        assert abs(score - base_score) < 1e-2


def test_block_reml_forward_result_does_not_depend_on_the_starting_point() -> None:
    """The starting point of the outer search cannot make the problem infeasible.

    ``init_rhos[1] = 28`` puts the first trial at ``lambda_2 = e^28 ~ 1.4e12``,
    the regime where the natural-parameterization normal matrix loses float64
    positive-definiteness. The search must retreat from that trial point, not
    abort. Every other rung of the grid already returns the same optimum.
    """
    designs, penalties, y, weights = _problem()

    reference = _fit(designs, penalties, y, weights)
    reference_fitted = np.asarray(reference["fitted"], dtype=np.float64).ravel()

    for rho_2 in (5.0, 10.0, 15.0, 20.0, 25.0, 28.0, 30.0, 32.0, 35.0, 40.0):
        init_rhos = np.array([-7.37, rho_2], dtype=np.float64)
        fit = _fit(designs, penalties, y, weights, init_rhos=init_rhos)
        fitted = np.asarray(fit["fitted"], dtype=np.float64).ravel()
        score = float(np.asarray(fit["reml_score"], dtype=np.float64))
        assert np.isfinite(fitted).all()
        assert np.isfinite(score)
        assert np.abs(fitted - reference_fitted).max() < 1e-4, (
            f"init_rhos[1]={rho_2} produced a different fit "
            f"({np.abs(fitted - reference_fitted).max():.3e})"
        )
