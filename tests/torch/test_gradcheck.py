"""Finite-difference gradient checks for every differentiable gamfit.torch primitive.

Each ``torch.autograd.Function`` in :mod:`gamfit.torch` wires a Rust analytic
backward into torch's gradient flow. ``torch.autograd.gradcheck`` confirms that
backward is consistent with finite differences taken through the forward.
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


# Conservative tolerances: f64 throughout, but the underlying solves are
# iterative so we leave the defaults (atol=1e-5, rtol=1e-3) loose enough to
# absorb fixed-point convergence noise.
_GRADCHECK_EPS: float = 1e-6
_GRADCHECK_ATOL: float = 1e-5
_GRADCHECK_RTOL: float = 1e-3
_GRADCHECK_NONDET_TOL: float = 1e-6


# ----------------------------- basis primitives ----------------------------- #


def test_bspline_basis_gradcheck() -> None:
    _require_ffi("bspline_basis")
    rng = np.random.default_rng(11)
    t = torch.tensor(
        rng.uniform(0.05, 0.95, size=8), dtype=torch.float64, requires_grad=True
    )
    knots = torch.tensor(np.linspace(0.0, 1.0, 9), dtype=torch.float64)

    def f(t_: torch.Tensor) -> torch.Tensor:
        return gt.bspline_basis(t_, knots, degree=3, periodic=False)

    assert torch.autograd.gradcheck(
        f,
        (t,),
        eps=_GRADCHECK_EPS,
        atol=_GRADCHECK_ATOL,
        rtol=_GRADCHECK_RTOL,
        nondet_tol=_GRADCHECK_NONDET_TOL,
    )


def test_duchon_basis_1d_gradcheck() -> None:
    _require_ffi("duchon_basis_1d")
    rng = np.random.default_rng(12)
    t = torch.tensor(
        rng.uniform(0.05, 0.95, size=8), dtype=torch.float64, requires_grad=True
    )
    centers = torch.tensor(np.linspace(0.0, 1.0, 5), dtype=torch.float64)

    def f(t_: torch.Tensor) -> torch.Tensor:
        return gt.duchon_basis_1d(t_, centers, m=2, periodic=False)

    assert torch.autograd.gradcheck(
        f,
        (t,),
        eps=_GRADCHECK_EPS,
        atol=_GRADCHECK_ATOL,
        rtol=_GRADCHECK_RTOL,
        nondet_tol=_GRADCHECK_NONDET_TOL,
    )


# ------------------------------ REML primitives ----------------------------- #


def _reml_inputs(
    n: int = 12, m: int = 3, d: int = 1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m))
    beta = rng.standard_normal((m, d))
    Y = X @ beta + 0.1 * rng.standard_normal((n, d))
    penalty = np.eye(m)
    return X, Y, penalty


def test_gaussian_reml_fit_gradcheck() -> None:
    _require_ffi("gaussian_reml_fit")
    X, Y, penalty = _reml_inputs(seed=13)
    x_t = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(Y, dtype=torch.float64, requires_grad=True)
    p_t = torch.tensor(penalty, dtype=torch.float64)

    def f(x_: torch.Tensor, y_: torch.Tensor) -> torch.Tensor:
        out = gt.gaussian_reml_fit(x_, y_, p_t)
        # Combine all four outputs into a single scalar so gradcheck can chase
        # any subset of the backward paths.
        return out.coefficients.sum() + out.fitted.sum() + out.lam.sum() + out.reml_score

    assert torch.autograd.gradcheck(
        f,
        (x_t, y_t),
        eps=_GRADCHECK_EPS,
        atol=_GRADCHECK_ATOL,
        rtol=_GRADCHECK_RTOL,
        nondet_tol=_GRADCHECK_NONDET_TOL,
    )


def test_gaussian_reml_fit_batched_gradcheck() -> None:
    _require_ffi("gaussian_reml_fit_batched")
    rng = np.random.default_rng(100)
    counts = [10, 11]
    offsets = np.cumsum([0] + counts).astype(np.uintp)
    n_total = int(offsets[-1])
    m, d = 3, 1
    X = rng.standard_normal((n_total, m))
    beta = rng.standard_normal((m, d))
    Y = X @ beta + 0.1 * rng.standard_normal((n_total, d))
    penalty = np.eye(m)
    x_t = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(Y, dtype=torch.float64, requires_grad=True)
    p_t = torch.tensor(penalty, dtype=torch.float64)
    off_t = torch.tensor(offsets)

    def f(x_: torch.Tensor, y_: torch.Tensor) -> torch.Tensor:
        out = gt.gaussian_reml_fit_batched(x_, y_, off_t, p_t)
        return out.coefficients.sum() + out.fitted.sum() + out.lam.sum() + out.reml_score.sum()

    assert torch.autograd.gradcheck(
        f,
        (x_t, y_t),
        eps=_GRADCHECK_EPS,
        atol=_GRADCHECK_ATOL,
        rtol=_GRADCHECK_RTOL,
        nondet_tol=_GRADCHECK_NONDET_TOL,
    )


def test_gaussian_reml_fit_positions_gradcheck() -> None:
    _require_ffi("gaussian_reml_fit_positions")
    rng = np.random.default_rng(15)
    n = 12
    t = np.sort(rng.uniform(0.05, 0.95, size=n))
    Y = (np.sin(2 * np.pi * t)).reshape(-1, 1) + 0.05 * rng.standard_normal((n, 1))
    knots = np.linspace(0.0, 1.0, 9)
    M = knots.size - 3 - 1
    penalty = np.eye(M)
    t_t = torch.tensor(t, dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(Y, dtype=torch.float64, requires_grad=True)
    k_t = torch.tensor(knots, dtype=torch.float64)
    p_t = torch.tensor(penalty, dtype=torch.float64)

    def f(t_: torch.Tensor, y_: torch.Tensor) -> torch.Tensor:
        out = gt.gaussian_reml_fit_positions(t_, y_, "bspline", k_t, p_t)
        return out.coefficients.sum() + out.fitted.sum() + out.lam.sum() + out.reml_score

    assert torch.autograd.gradcheck(
        f,
        (t_t, y_t),
        eps=_GRADCHECK_EPS,
        atol=_GRADCHECK_ATOL,
        rtol=_GRADCHECK_RTOL,
        nondet_tol=_GRADCHECK_NONDET_TOL,
    )


def test_gaussian_reml_fit_positions_batched_gradcheck() -> None:
    _require_ffi("gaussian_reml_fit_positions_batched")
    rng = np.random.default_rng(16)
    counts = [10, 12]
    offsets = np.cumsum([0] + counts).astype(np.uintp)
    n_total = int(offsets[-1])
    t = np.concatenate(
        [np.sort(rng.uniform(0.05, 0.95, size=c)) for c in counts]
    )
    Y = (np.sin(2 * np.pi * t)).reshape(-1, 1) + 0.05 * rng.standard_normal((n_total, 1))
    knots = np.linspace(0.0, 1.0, 9)
    M = knots.size - 3 - 1
    penalty = np.eye(M)
    t_t = torch.tensor(t, dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(Y, dtype=torch.float64, requires_grad=True)
    k_t = torch.tensor(knots, dtype=torch.float64)
    p_t = torch.tensor(penalty, dtype=torch.float64)
    off_t = torch.tensor(offsets)

    def f(t_: torch.Tensor, y_: torch.Tensor) -> torch.Tensor:
        out = gt.gaussian_reml_fit_positions_batched(t_, y_, off_t, "bspline", k_t, p_t)
        return out.coefficients.sum() + out.fitted.sum() + out.lam.sum() + out.reml_score.sum()

    assert torch.autograd.gradcheck(
        f,
        (t_t, y_t),
        eps=_GRADCHECK_EPS,
        atol=_GRADCHECK_ATOL,
        rtol=_GRADCHECK_RTOL,
        nondet_tol=_GRADCHECK_NONDET_TOL,
    )
