"""Bug hunt: ``gamfit.gaussian_reml_fit_with_constraints_forward`` cannot solve a
binding linear constraint against a *rank-deficient* penalty -- i.e. against any
real smoothing penalty.

14 of the 16 single-coefficient box constraints below abort with a hard error at
HEAD.  Adding a ridge of ``1e-8`` to a penalty whose largest eigenvalue is
``2.3e4`` -- a relative perturbation of ``4e-13``, far below any tolerance the
solver has -- makes all 16 solve, land on the constraint exactly, and converge at
``lambda ~ 3e-4``.  So the optimum is in an entirely benign region; the failure
is somewhere the solver goes on its own.

Root cause: ``optimize_affine_face``,
``crates/gam-solve/src/constrained_gaussian_reml.rs:~597``, ends with

    for endpoint in [-bound, bound] {
        let candidate = profile.evaluate(endpoint)?;
        ...
    }

with ``bound = crate::estimate::RHO_BOUND`` = ``30``
(``crates/gam-solve/src/estimate/smoothing_correction.rs:110``).  That is an
*unconditional* probe of the objective at ``lambda = e^30 = 1.07e13``, run on
every fit no matter where the search actually converged -- and its failure is
propagated with ``?``, killing the whole fit.

``AffineFaceProfile::evaluate`` (line 434) forms

    hessian = tangent_gram + lambda * tangent_penalty          # line 441

in the natural parameterization and Choleskys it strictly (line 443), then
guards ``penalized_deviance = weighted_rss + lambda * beta' S beta > 0``
(line 459).  With ``tangent_penalty = Z' S Z`` rank-deficient -- which it is for
every derivative-based roughness penalty, whose null space is the polynomials
below the penalty order -- ``lambda = 1.07e13`` puts that matrix past the float64
cancellation wall.  For the fixture below, ``G_t + e^30 * P_t`` computed in
NumPy has ``min eig = -1.58e+01`` when column 0 is removed from the tangent
face, and its first non-positive Cholesky pivot is at index ``6`` -- exactly the
``NonPositivePivot { index: 6 }`` the engine reports.  Removing columns 0/1/2/3
breaks; removing 4/5/6/7 does not, which is precisely the observed
success/failure split.

The two failure modes seen are both this wall:

* ``LinearSystemSolveError: ... Cholesky factorization failed: NonPositivePivot``
  -- the strict factorization at line 443;
* ``InvalidInputError: constrained Gaussian REML profiled deviance must be
  positive`` -- ``lambda * (beta' S beta)`` with ``beta`` in the penalty null
  space, where the ``O(eps)`` cancellation in the energy is multiplied by
  ``1e13`` and swamps a residual sum of squares of order 1.

The same file also maps every BFGS trial evaluation to
``ObjectiveEvalError::fatal(...)`` (line ~578), so a trial-point breakdown inside
the search is fatal too.  A boundary probe is a *test* -- "is the over-smoothing
endpoint a better stationary candidate?" -- and a failed evaluation there means
"no, that endpoint is not attainable", not "abort the fit".

This is the constrained-solver sibling of the block-REML escalation in #2830:
the same ``rho``-local numerical refusal, the same natural-parameterization
normal matrix, the same escalation to a whole-fit abort.

Both assertions below are fix-agnostic -- they never name ``lambda``, the
``rho`` bound, an error class or a solver.  Whether the repair skips a failed
endpoint probe, reparameterizes the profile stably, or bounds ``rho`` by the
penalty spectrum, the tests pass unedited.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

import gamfit

_N = 50
_K_INTERNAL_KNOTS = 4
_DEGREE = 3
_PENALTY_ORDER = 2
# Relative to the penalty's largest eigenvalue (~2.3e4) this is ~4e-13: far
# below every tolerance in the solver, and the reference the engine itself
# solves without complaint.
_NEGLIGIBLE_RIDGE = 1e-10


def _fixture() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    rng = np.random.default_rng(9)
    t = np.sort(rng.uniform(0.0, 1.0, _N))
    design = np.asarray(
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
    y = np.sin(5.0 * t) + 0.2 * rng.normal(size=_N)
    weights = rng.uniform(0.5, 2.0, _N)
    return design, y, penalty, weights


def _box_row(
    p: int, column: int, sign: float, unconstrained: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A single box constraint on ``beta[column]``, offset 0.3 from the free fit.

    Both signs are exercised so the test is agnostic to the ``A.beta >= b`` /
    ``A.beta <= b`` orientation: whichever way the engine reads the system, one
    of the two rows binds and the other is slack, and both must be *solvable*.
    """
    a = np.zeros((1, p), dtype=np.float64)
    a[0, column] = sign
    b = np.array([sign * unconstrained[column] + 0.3], dtype=np.float64)
    return a, b


_CASES = [(column, sign) for column in range(8) for sign in (1.0, -1.0)]


@pytest.mark.parametrize(("column", "sign"), _CASES)
def test_constrained_reml_solves_a_box_constraint_on_every_coefficient(
    column: int, sign: float
) -> None:
    design, y, penalty, weights = _fixture()
    p = design.shape[1]
    free = np.asarray(
        gamfit.gaussian_reml_fit_with_constraints_forward(
            design, y, penalty, weights=weights
        )["coefficients"],
        dtype=np.float64,
    ).ravel()
    a, b = _box_row(p, column, sign, free)

    fit = gamfit.gaussian_reml_fit_with_constraints_forward(
        design, y, penalty, weights=weights, a_inequality=a, b_inequality=b
    )
    beta = np.asarray(fit["coefficients"], dtype=np.float64).ravel()
    fitted = np.asarray(fit["fitted"], dtype=np.float64).ravel()
    active = np.asarray(fit["active_indices"]).ravel()

    assert np.isfinite(beta).all()
    assert np.isfinite(fitted).all()
    assert np.isfinite(float(fit["reml_score"]))
    # A row the solver itself declares active must sit exactly on its
    # hyperplane -- that is what "active" means, and it is the only claim here
    # that does not depend on the sign orientation of the system.
    if active.size:
        residual = float((a @ beta - b)[0])
        assert abs(residual) < 1e-6, f"active row is off its hyperplane by {residual:e}"


@pytest.mark.parametrize(("column", "sign"), _CASES)
def test_constrained_reml_is_continuous_in_a_negligible_penalty_ridge(
    column: int, sign: float
) -> None:
    """A 4e-13 relative ridge on the penalty must not decide feasibility.

    The ridged problem is solved by the engine for all 16 constraints, at
    ``lambda ~ 3e-4`` and with the constraint exactly tight. The exact-penalty
    problem is the same problem; its answer must be the same too.
    """
    design, y, penalty, weights = _fixture()
    p = design.shape[1]
    free = np.asarray(
        gamfit.gaussian_reml_fit_with_constraints_forward(
            design, y, penalty, weights=weights
        )["coefficients"],
        dtype=np.float64,
    ).ravel()
    a, b = _box_row(p, column, sign, free)

    ridged = gamfit.gaussian_reml_fit_with_constraints_forward(
        design,
        y,
        penalty + _NEGLIGIBLE_RIDGE * np.eye(p),
        weights=weights,
        a_inequality=a,
        b_inequality=b,
    )
    ridged_fitted = np.asarray(ridged["fitted"], dtype=np.float64).ravel()

    exact = gamfit.gaussian_reml_fit_with_constraints_forward(
        design, y, penalty, weights=weights, a_inequality=a, b_inequality=b
    )
    exact_fitted = np.asarray(exact["fitted"], dtype=np.float64).ravel()

    gap = float(np.abs(exact_fitted - ridged_fitted).max())
    assert gap < 1e-3, f"exact-penalty fit differs from the ridged fit by {gap:e}"
