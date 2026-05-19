"""Finite-difference gradient checks for the ``grad_penalty`` autograd path.

The four ``gaussian_reml_fit*`` primitives carry an analytic backward through
the penalty matrix ``S``. The companion checks in :mod:`test_gradcheck` keep
``S`` frozen and only check gradients on ``(X, Y, t)``; here we set
``requires_grad=True`` on the penalty input so
:func:`torch.autograd.gradcheck` exercises the analytic ``grad_penalty``
against a finite-difference reference.

``torch.autograd.gradcheck`` perturbs each entry of ``S`` independently — that
momentarily breaks symmetry. The Rust solver handles this directly: it
canonicalizes the input to ``0.5 (S + Sᵀ)`` on entry and returns a symmetric
``grad_penalty`` matching that chain rule. No Python-side symmetrization is
needed in user code or in the tests.

If a check fires, gradcheck reports per-entry mismatches so the failing
helper can be localised.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    import gamfit.torch as gt
except ImportError:
    pytest.skip("torch dependency unavailable", allow_module_level=True)


def _require_ffi(name: str) -> None:
    from gamfit._binding import rust_module

    if not hasattr(rust_module(), name):
        pytest.skip(f"engine missing FFI export `{name}`")


_GRADCHECK_EPS: float = 1e-6
_GRADCHECK_ATOL: float = 1e-4
_GRADCHECK_RTOL: float = 1e-3
_GRADCHECK_NONDET_TOL: float = 1e-6


def _full_rank_symmetric_penalty(p: int, seed: int) -> np.ndarray:
    """Random symmetric positive-definite penalty.

    Used wherever ``gradcheck`` perturbs individual entries of ``S``: the
    diagonal shift keeps the smallest eigenvalue comfortably above gamfit's
    PSD tolerance under single-entry perturbations of size ``eps``, so the
    forward never trips its ill-conditioning rejection during FD.

    Rank-deficient ``S`` — the production regime — is exercised separately
    in :mod:`test_gaussian_reml_edf_backward` via directional FD along
    perturbations projected onto ``range(S)``. Per-entry gradcheck on a
    rank-deficient base would inherently push the null-space eigenvalue
    through zero on individual coordinate perturbations and is the wrong
    tool for that regime.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((p, p))
    return (0.5 * (raw + raw.T) + float(p) * np.eye(p)).astype(np.float64)


def _reml_inputs(
    n: int = 12, m: int = 3, d: int = 1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m))
    beta = rng.standard_normal((m, d))
    Y = X @ beta + 0.1 * rng.standard_normal((n, d))
    return X, Y, _full_rank_symmetric_penalty(m, seed=seed + 1000)


# --------------------------- single REML fit ---------------------------- #


def test_gaussian_reml_fit_penalty_gradcheck() -> None:
    """``grad_penalty`` must match finite-difference for the single-fit primitive."""
    _require_ffi("gaussian_reml_fit")
    X, Y, penalty = _reml_inputs(seed=130)
    x_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(Y, dtype=torch.float64)
    p_t = torch.tensor(penalty, dtype=torch.float64, requires_grad=True)

    def f(p_: torch.Tensor) -> torch.Tensor:
        out = gt.gaussian_reml_fit(x_t, y_t, p_)
        return (
            out.coefficients.sum()
            + out.fitted.sum()
            + out.lam.sum()
            + out.reml_score
            + out.edf
        )

    assert torch.autograd.gradcheck(
        f,
        (p_t,),
        eps=_GRADCHECK_EPS,
        atol=_GRADCHECK_ATOL,
        rtol=_GRADCHECK_RTOL,
        nondet_tol=_GRADCHECK_NONDET_TOL,
    )


def test_gaussian_reml_fit_penalty_gradcheck_coefficients_only() -> None:
    """Isolate the upstream-``β`` path through ``add_ridge_profile_vjp``."""
    _require_ffi("gaussian_reml_fit")
    X, Y, penalty = _reml_inputs(seed=131)
    x_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(Y, dtype=torch.float64)
    p_t = torch.tensor(penalty, dtype=torch.float64, requires_grad=True)

    def f(p_: torch.Tensor) -> torch.Tensor:
        return gt.gaussian_reml_fit(x_t, y_t, p_).coefficients.sum()

    assert torch.autograd.gradcheck(
        f, (p_t,),
        eps=_GRADCHECK_EPS, atol=_GRADCHECK_ATOL, rtol=_GRADCHECK_RTOL,
        nondet_tol=_GRADCHECK_NONDET_TOL,
    )


def test_gaussian_reml_fit_penalty_gradcheck_reml_score_only() -> None:
    """Isolate the upstream-``R`` path through ``add_reml_score_vjp``."""
    _require_ffi("gaussian_reml_fit")
    X, Y, penalty = _reml_inputs(seed=132)
    x_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(Y, dtype=torch.float64)
    p_t = torch.tensor(penalty, dtype=torch.float64, requires_grad=True)

    def f(p_: torch.Tensor) -> torch.Tensor:
        return gt.gaussian_reml_fit(x_t, y_t, p_).reml_score

    assert torch.autograd.gradcheck(
        f, (p_t,),
        eps=_GRADCHECK_EPS, atol=_GRADCHECK_ATOL, rtol=_GRADCHECK_RTOL,
        nondet_tol=_GRADCHECK_NONDET_TOL,
    )


def test_gaussian_reml_fit_penalty_gradcheck_edf_only() -> None:
    """Isolate the upstream-``edf`` path through ``add_edf_vjp``."""
    _require_ffi("gaussian_reml_fit")
    X, Y, penalty = _reml_inputs(seed=133)
    x_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(Y, dtype=torch.float64)
    p_t = torch.tensor(penalty, dtype=torch.float64, requires_grad=True)

    def f(p_: torch.Tensor) -> torch.Tensor:
        return gt.gaussian_reml_fit(x_t, y_t, p_).edf

    assert torch.autograd.gradcheck(
        f, (p_t,),
        eps=_GRADCHECK_EPS, atol=_GRADCHECK_ATOL, rtol=_GRADCHECK_RTOL,
        nondet_tol=_GRADCHECK_NONDET_TOL,
    )


def test_gaussian_reml_fit_penalty_gradcheck_lambda_only() -> None:
    """Isolate the upstream-``λ`` path through the implicit-function chain."""
    _require_ffi("gaussian_reml_fit")
    X, Y, penalty = _reml_inputs(seed=134)
    x_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(Y, dtype=torch.float64)
    p_t = torch.tensor(penalty, dtype=torch.float64, requires_grad=True)

    def f(p_: torch.Tensor) -> torch.Tensor:
        return gt.gaussian_reml_fit(x_t, y_t, p_).lam.sum()

    assert torch.autograd.gradcheck(
        f, (p_t,),
        eps=_GRADCHECK_EPS, atol=_GRADCHECK_ATOL, rtol=_GRADCHECK_RTOL,
        nondet_tol=_GRADCHECK_NONDET_TOL,
    )


# --------------------------- batched REML fit --------------------------- #


def test_gaussian_reml_fit_batched_penalty_gradcheck() -> None:
    _require_ffi("gaussian_reml_fit_batched")
    rng = np.random.default_rng(200)
    counts = [10, 11]
    offsets = np.cumsum([0] + counts).astype(np.uintp)
    n_total = int(offsets[-1])
    m, d = 3, 1
    X = rng.standard_normal((n_total, m))
    beta = rng.standard_normal((m, d))
    Y = X @ beta + 0.1 * rng.standard_normal((n_total, d))
    penalty = _full_rank_symmetric_penalty(m, seed=201)

    x_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(Y, dtype=torch.float64)
    off_t = torch.tensor(offsets)
    p_t = torch.tensor(penalty, dtype=torch.float64, requires_grad=True)

    def f(p_: torch.Tensor) -> torch.Tensor:
        out = gt.gaussian_reml_fit_batched(x_t, y_t, off_t, p_)
        return (
            out.coefficients.sum()
            + out.fitted.sum()
            + out.lam.sum()
            + out.reml_score.sum()
            + out.edf.sum()
        )

    assert torch.autograd.gradcheck(
        f, (p_t,),
        eps=_GRADCHECK_EPS, atol=_GRADCHECK_ATOL, rtol=_GRADCHECK_RTOL,
        nondet_tol=_GRADCHECK_NONDET_TOL,
    )


# --------------------------- positions REML fit ------------------------- #


def test_gaussian_reml_fit_positions_penalty_gradcheck() -> None:
    _require_ffi("gaussian_reml_fit_positions")
    rng = np.random.default_rng(300)
    n = 12
    t = np.sort(rng.uniform(0.05, 0.95, size=n))
    Y = (np.sin(2 * np.pi * t)).reshape(-1, 1) + 0.05 * rng.standard_normal((n, 1))
    knots = np.linspace(0.0, 1.0, 9)
    m = knots.size - 3 - 1
    penalty = _full_rank_symmetric_penalty(m, seed=301)

    t_t = torch.tensor(t, dtype=torch.float64)
    y_t = torch.tensor(Y, dtype=torch.float64)
    k_t = torch.tensor(knots, dtype=torch.float64)
    p_t = torch.tensor(penalty, dtype=torch.float64, requires_grad=True)

    def f(p_: torch.Tensor) -> torch.Tensor:
        out = gt.gaussian_reml_fit_positions(t_t, y_t, "bspline", k_t, p_)
        return (
            out.coefficients.sum()
            + out.fitted.sum()
            + out.lam.sum()
            + out.reml_score
            + out.edf
        )

    assert torch.autograd.gradcheck(
        f, (p_t,),
        eps=_GRADCHECK_EPS, atol=_GRADCHECK_ATOL, rtol=_GRADCHECK_RTOL,
        nondet_tol=_GRADCHECK_NONDET_TOL,
    )


# ----------------------- batched positions REML fit --------------------- #


def test_gaussian_reml_fit_positions_batched_penalty_gradcheck() -> None:
    _require_ffi("gaussian_reml_fit_positions_batched")
    rng = np.random.default_rng(400)
    counts = [10, 12]
    offsets = np.cumsum([0] + counts).astype(np.uintp)
    n_total = int(offsets[-1])
    t = np.concatenate(
        [np.sort(rng.uniform(0.05, 0.95, size=c)) for c in counts]
    )
    Y = (np.sin(2 * np.pi * t)).reshape(-1, 1) + 0.05 * rng.standard_normal((n_total, 1))
    knots = np.linspace(0.0, 1.0, 9)
    m = knots.size - 3 - 1
    penalty = _full_rank_symmetric_penalty(m, seed=401)

    t_t = torch.tensor(t, dtype=torch.float64)
    y_t = torch.tensor(Y, dtype=torch.float64)
    k_t = torch.tensor(knots, dtype=torch.float64)
    off_t = torch.tensor(offsets)
    p_t = torch.tensor(penalty, dtype=torch.float64, requires_grad=True)

    def f(p_: torch.Tensor) -> torch.Tensor:
        out = gt.gaussian_reml_fit_positions_batched(
            t_t, y_t, off_t, "bspline", k_t, p_
        )
        return (
            out.coefficients.sum()
            + out.fitted.sum()
            + out.lam.sum()
            + out.reml_score.sum()
            + out.edf.sum()
        )

    assert torch.autograd.gradcheck(
        f, (p_t,),
        eps=_GRADCHECK_EPS, atol=_GRADCHECK_ATOL, rtol=_GRADCHECK_RTOL,
        nondet_tol=_GRADCHECK_NONDET_TOL,
    )
