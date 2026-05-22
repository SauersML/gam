"""Finite-difference gradient checks for every differentiable gamfit.torch primitive.

Each ``torch.autograd.Function`` in :mod:`gamfit.torch` wires a Rust analytic
backward into torch's gradient flow. ``torch.autograd.gradcheck`` confirms that
backward is consistent with finite differences taken through the forward.
"""

from __future__ import annotations

from typing import cast

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
        return cast(torch.Tensor, gt.bspline_basis(t_, knots, degree=3, periodic=False))

    assert torch.autograd.gradcheck(
        f,
        (t,),
        eps=_GRADCHECK_EPS,
        atol=_GRADCHECK_ATOL,
        rtol=_GRADCHECK_RTOL,
        nondet_tol=_GRADCHECK_NONDET_TOL,
    )


def test_duchon_basis_gradcheck() -> None:
    _require_ffi("duchon_basis_1d")
    rng = np.random.default_rng(12)
    t = torch.tensor(
        rng.uniform(0.05, 0.95, size=8), dtype=torch.float64, requires_grad=True
    )
    centers = torch.tensor(np.linspace(0.0, 1.0, 5), dtype=torch.float64)

    def f(t_: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, gt.duchon_basis(t_, centers, m=2))

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
    if m > 2:
        penalty[2, 1] = 0.08
        penalty[1, 2] = 0.08
    return X, Y, penalty


def test_gaussian_reml_fit_gradcheck() -> None:
    _require_ffi("gaussian_reml_fit")
    X, Y, penalty = _reml_inputs(seed=13)
    x_t = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(Y, dtype=torch.float64, requires_grad=True)
    p_t = torch.tensor(penalty, dtype=torch.float64)

    def f(x_: torch.Tensor, y_: torch.Tensor, p_: torch.Tensor) -> torch.Tensor:
        out = gt.gaussian_reml_fit(x_, y_, p_)
        # Combine all four outputs into a single scalar so gradcheck can chase
        # any subset of the backward paths.
        return out.coefficients.sum() + out.fitted.sum() + out.lam.sum() + out.reml_score

    assert torch.autograd.gradcheck(
        f,
        (x_t, y_t, p_t),
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
    penalty[2, 1] = 0.06
    penalty[1, 2] = 0.06
    x_t = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(Y, dtype=torch.float64, requires_grad=True)
    p_t = torch.tensor(penalty, dtype=torch.float64)
    off_t = torch.tensor(offsets)

    def f(x_: torch.Tensor, y_: torch.Tensor, p_: torch.Tensor) -> torch.Tensor:
        out = gt.gaussian_reml_fit_batched(x_, y_, off_t, p_)
        return out.coefficients.sum() + out.fitted.sum() + out.lam.sum() + out.reml_score.sum()

    assert torch.autograd.gradcheck(
        f,
        (x_t, y_t, p_t),
        eps=_GRADCHECK_EPS,
        atol=_GRADCHECK_ATOL,
        rtol=_GRADCHECK_RTOL,
        nondet_tol=_GRADCHECK_NONDET_TOL,
    )


def _reml_block_inputs(
    n: int = 18, p_per: int = 3, seed: int = 20260522
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal((n, p_per))
    x2 = rng.standard_normal((n, p_per))
    beta_true = rng.standard_normal((2 * p_per,))
    y = np.concatenate([x1, x2], axis=1) @ beta_true + 0.2 * rng.standard_normal(n)
    s1 = (
        np.eye(p_per)
        + 0.05 * np.diag(np.ones(p_per - 1), 1)
        + 0.05 * np.diag(np.ones(p_per - 1), -1)
    )
    s2 = (
        np.eye(p_per)
        + 0.03 * np.diag(np.ones(p_per - 1), 1)
        + 0.03 * np.diag(np.ones(p_per - 1), -1)
    )
    weights = rng.uniform(0.6, 1.4, n)
    return x1, x2, s1, s2, y, weights


def _reml_block_tensors() -> tuple[torch.Tensor, ...]:
    x1, x2, s1, s2, y, weights = _reml_block_inputs()
    return (
        torch.tensor(x1, dtype=torch.float64, requires_grad=True),
        torch.tensor(x2, dtype=torch.float64, requires_grad=True),
        torch.tensor(s1, dtype=torch.float64, requires_grad=True),
        torch.tensor(s2, dtype=torch.float64, requires_grad=True),
        torch.tensor(y[:, None], dtype=torch.float64, requires_grad=True),
        torch.tensor(weights, dtype=torch.float64, requires_grad=True),
    )


def test_gaussian_reml_fit_blocks_reml_score_gradcheck() -> None:
    """The negative REML score VJP has the same sign as profile perturbations."""
    _require_ffi("gaussian_reml_fit_blocks_forward")
    _require_ffi("gaussian_reml_fit_blocks_backward")
    x1_t, x2_t, s1_t, s2_t, y_t, w_t = _reml_block_tensors()

    def f(
        a: torch.Tensor,
        b: torch.Tensor,
        sa: torch.Tensor,
        sb: torch.Tensor,
        yy: torch.Tensor,
        ww: torch.Tensor,
    ) -> torch.Tensor:
        return gt.gaussian_reml_fit_blocks([a, b], [sa, sb], yy, weights=ww).reml_score

    assert torch.autograd.gradcheck(
        f,
        (x1_t, x2_t, s1_t, s2_t, y_t, w_t),
        eps=1e-5,
        atol=1e-4,
        rtol=1e-3,
        nondet_tol=1e-6,
    )


def test_gaussian_reml_fit_blocks_gradcheck() -> None:
    """Multi-block per-smooth-λ Gaussian REML backward matches FD.

    Drives ``gradcheck`` through the closed-form analytic VJP wired into the
    pyffi ``gaussian_reml_fit_blocks_backward`` entrypoint. Two design blocks
    plus their penalties, response, and row weights are all marked
    ``requires_grad=True`` so every input-side leg of the analytic backward
    (designs, penalties, y, weights) is exercised against central finite
    differences taken through the warm-started outer optimum.
    """
    _require_ffi("gaussian_reml_fit_blocks_forward")
    _require_ffi("gaussian_reml_fit_blocks_backward")
    x1_t, x2_t, s1_t, s2_t, y_t, w_t = _reml_block_tensors()

    def f(
        a: torch.Tensor,
        b: torch.Tensor,
        sa: torch.Tensor,
        sb: torch.Tensor,
        yy: torch.Tensor,
        ww: torch.Tensor,
    ) -> torch.Tensor:
        out = gt.gaussian_reml_fit_blocks([a, b], [sa, sb], yy, weights=ww)
        return (
            out.fitted.sum()
            + out.lambdas.sum()
            + out.log_lambdas.sum()
            + out.reml_score
            + out.edf.sum()
            + sum(c.sum() for c in out.coefficients)
        )

    assert torch.autograd.gradcheck(
        f,
        (x1_t, x2_t, s1_t, s2_t, y_t, w_t),
        eps=1e-5,
        atol=1e-4,
        rtol=1e-3,
        nondet_tol=1e-6,
    )


def test_gaussian_reml_fit_blocks_f1_matches_single_block_backward() -> None:
    """F=1 reduction sanity check.

    The multi-block backward at ``F=1`` must produce gradients consistent
    with the single-smooth ``gaussian_reml_fit`` analytic VJP. Both routes
    solve the same closed-form ridge at one ρ\\*, so differentiating
    ``fitted.sum()`` back to ``X`` should agree to inner convergence
    tolerance.
    """
    _require_ffi("gaussian_reml_fit_blocks_forward")
    _require_ffi("gaussian_reml_fit")

    rng = np.random.default_rng(2026)
    n, m = 30, 5
    X = rng.standard_normal((n, m))
    y = rng.standard_normal((n, 1))
    s = np.eye(m)

    x_blk = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    p_blk = torch.tensor(s, dtype=torch.float64)
    y_blk = torch.tensor(y, dtype=torch.float64)
    out_blocks = gt.gaussian_reml_fit_blocks([x_blk], [p_blk], y_blk)
    out_blocks.fitted.sum().backward()
    assert x_blk.grad is not None
    grad_x_blocks = x_blk.grad.detach().clone()

    x_single = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    p_single = torch.tensor(s, dtype=torch.float64)
    y_single = torch.tensor(y, dtype=torch.float64)
    out_single = gt.gaussian_reml_fit(x_single, y_single, p_single)
    out_single.fitted.sum().backward()
    assert x_single.grad is not None
    grad_x_single = x_single.grad.detach().clone()

    torch.testing.assert_close(grad_x_blocks, grad_x_single, rtol=2e-3, atol=2e-4)
