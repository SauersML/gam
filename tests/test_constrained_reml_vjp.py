"""Gradient checks for the constrained Gaussian REML analytic VJP.

The constrained single-block Gaussian REML fit
(:func:`gamfit.torch._reml.gaussian_reml_fit_with_constraints`) carries an
exact analytic backward in BOTH regimes:

* **Interior cert** (the inequality constraints are present but strictly
  inactive at the optimum): the envelope theorem applies in full ``p``-space
  and the VJP is the closed-form Gaussian REML backward.
* **Active cert** (at least one inequality binds, ``A_act β̂ = 0``): the VJP is
  the tangent-projected backward in the ``Z = null(A_act)`` reduction, i.e.
  ``H⁻¹ → Z(ZᵀHZ)⁻¹Zᵀ`` and ``S⁺ → Z(ZᵀSZ)⁺Zᵀ``.

Both regimes are validated here with ``torch.autograd.gradcheck`` on the free
inputs ``(x, y, penalty, weights)`` at float64. The active case previously
raised ``NotImplementedError`` from the Rust binding; it must now pass.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gamfit.torch._reml import gaussian_reml_fit_with_constraints  # noqa: E402


def _require_ffi() -> None:
    from gamfit._binding import rust_module

    if not hasattr(rust_module(), "gaussian_reml_fit_with_constraints_forward"):
        pytest.skip("engine missing FFI export `gaussian_reml_fit_with_constraints_*`")


# float64 gradcheck defaults: principled, not weakened. The closed-form
# Gaussian REML solve is a direct factorisation (no fixed-point), so the
# analytic VJP matches central differences to tight tolerance.
_EPS = 1e-6
_ATOL = 1e-5
_RTOL = 1e-3
_NONDET_TOL = 1e-6


def _curvature_design(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic design ``[1, t, t²]`` on an evenly spaced grid."""
    t = np.linspace(-1.0, 1.0, n)
    x = np.column_stack([np.ones_like(t), t, t * t])
    return x, t


# Ridge on slope + curvature, intercept unpenalised. Inside the tangent space
# of the curvature constraint (``Z = span{e0, e1}``) this restricts to a
# rank-1 ridge on the slope, so the reduced REML problem is well posed.
_PENALTY = np.diag([0.0, 1.0, 1.0])

# Curvature constraint ``[0, 0, 1] · β ≥ 0`` with a zero bound. The backward
# supports zero-bound certificates; the active set is then ``curvature = 0``.
_A_INEQ = np.array([[0.0, 0.0, 1.0]])
_B_INEQ = np.array([0.0])


def _scalar_objective(out: object) -> "torch.Tensor":
    """Collapse the differentiable forward outputs into one scalar.

    Exercises every differentiable upstream path (coefficients, fitted,
    reml_score, edf, log_lambda) with linearly independent weights so the
    full VJP is checked, not just one component.
    """
    o = cast("torch.Tensor", out.coefficients).reshape(-1)  # type: ignore[attr-defined]
    f = cast("torch.Tensor", out.fitted).reshape(-1)  # type: ignore[attr-defined]
    weights_c = torch.arange(1, o.numel() + 1, dtype=torch.float64)
    weights_f = torch.linspace(0.5, 1.5, f.numel(), dtype=torch.float64)
    return (
        (o * weights_c).sum()
        + 0.7 * (f * weights_f).sum()
        + 1.3 * cast("torch.Tensor", out.reml_score)  # type: ignore[attr-defined]
        + 0.4 * cast("torch.Tensor", out.edf)  # type: ignore[attr-defined]
        + 0.9 * cast("torch.Tensor", out.log_lambda)  # type: ignore[attr-defined]
    )


def _make_inputs(
    y_np: np.ndarray, x_np: np.ndarray
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    n = x_np.shape[0]
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(y_np.reshape(-1, 1), dtype=torch.float64, requires_grad=True)
    penalty = torch.tensor(_PENALTY, dtype=torch.float64, requires_grad=True)
    weights = torch.tensor(
        np.linspace(0.6, 1.4, n), dtype=torch.float64, requires_grad=True
    )
    return x, y, penalty, weights


def _forward(
    x: "torch.Tensor",
    y: "torch.Tensor",
    penalty: "torch.Tensor",
    weights: "torch.Tensor",
) -> object:
    a = torch.tensor(_A_INEQ, dtype=torch.float64)
    b = torch.tensor(_B_INEQ, dtype=torch.float64)
    return gaussian_reml_fit_with_constraints(
        x,
        y,
        penalty,
        weights=weights,
        a_inequality=a,
        b_inequality=b,
    )


def test_constrained_reml_vjp_interior_cert() -> None:
    """Constraint present but inactive: gradcheck the full-space envelope VJP."""
    _require_ffi()
    n = 24
    x_np, t = _curvature_design(n)
    rng = np.random.default_rng(20240602)
    # Convex truth (positive curvature) → the `curvature ≥ 0` constraint is
    # satisfied strictly, so the cert is interior.
    y_np = 0.3 + 0.5 * t + 1.5 * t * t + 0.02 * rng.standard_normal(n)

    x, y, penalty, weights = _make_inputs(y_np, x_np)
    out = _forward(x, y, penalty, weights)
    assert (
        cast("torch.Tensor", out.active_indices).numel() == 0  # type: ignore[attr-defined]
    ), "expected an interior cert (no active constraint) for the convex truth"

    def f(
        x_: "torch.Tensor",
        y_: "torch.Tensor",
        p_: "torch.Tensor",
        w_: "torch.Tensor",
    ) -> "torch.Tensor":
        return _scalar_objective(_forward(x_, y_, p_, w_))

    assert torch.autograd.gradcheck(
        f,
        (x, y, penalty, weights),
        eps=_EPS,
        atol=_ATOL,
        rtol=_RTOL,
        nondet_tol=_NONDET_TOL,
    )


def test_constrained_reml_vjp_interior_cert_nonzero_slack_bound() -> None:
    """Interior cert with a NON-ZERO bound on a never-binding constraint.

    Regression for the backward's zero-bound guard being scoped to the active
    face: the curvature constraint ``[0,0,1]·β ≥ -1`` carries a non-zero bound
    but is strictly slack for the convex truth (curvature ≈ +1.5 ≥ -1), so the
    cert is interior and the backward is the full-space envelope VJP that never
    reads ``b``. Before the fix the top-level guard rejected *any* non-zero
    ``b_inequality`` with "supports only zero-bound inequality certificates",
    so this forward+backward raised instead of running; it must now gradcheck
    exactly like the zero-bound interior case.
    """
    _require_ffi()
    n = 24
    x_np, t = _curvature_design(n)
    rng = np.random.default_rng(20240602)
    # Convex truth (positive curvature) → `curvature ≥ -1` is satisfied
    # strictly, so the cert stays interior even with the non-zero bound.
    y_np = 0.3 + 0.5 * t + 1.5 * t * t + 0.02 * rng.standard_normal(n)

    def forward_slack(
        x_: "torch.Tensor",
        y_: "torch.Tensor",
        p_: "torch.Tensor",
        w_: "torch.Tensor",
    ) -> object:
        a = torch.tensor(_A_INEQ, dtype=torch.float64)
        b = torch.tensor([-1.0], dtype=torch.float64)  # non-zero, never binds
        return gaussian_reml_fit_with_constraints(
            x_, y_, p_, weights=w_, a_inequality=a, b_inequality=b
        )

    x, y, penalty, weights = _make_inputs(y_np, x_np)
    out = forward_slack(x, y, penalty, weights)
    assert (
        cast("torch.Tensor", out.active_indices).numel() == 0  # type: ignore[attr-defined]
    ), "expected an interior cert (constraint slack) despite the non-zero bound"

    def f(
        x_: "torch.Tensor",
        y_: "torch.Tensor",
        p_: "torch.Tensor",
        w_: "torch.Tensor",
    ) -> "torch.Tensor":
        return _scalar_objective(forward_slack(x_, y_, p_, w_))

    assert torch.autograd.gradcheck(
        f,
        (x, y, penalty, weights),
        eps=_EPS,
        atol=_ATOL,
        rtol=_RTOL,
        nondet_tol=_NONDET_TOL,
    )


def test_constrained_reml_vjp_active_cert() -> None:
    """Constraint binds at the optimum: gradcheck the tangent-projected VJP."""
    _require_ffi()
    n = 24
    x_np, t = _curvature_design(n)
    rng = np.random.default_rng(7)
    # Concave truth (negative curvature) → the unconstrained REML fit wants
    # curvature < 0, so the `curvature ≥ 0` constraint binds: the active-set
    # solver pins curvature to exactly 0 and the cert is active.
    y_np = 0.2 - 0.4 * t - 1.8 * t * t + 0.02 * rng.standard_normal(n)

    x, y, penalty, weights = _make_inputs(y_np, x_np)
    out = _forward(x, y, penalty, weights)
    active = cast("torch.Tensor", out.active_indices)  # type: ignore[attr-defined]
    assert active.numel() >= 1, (
        "expected the curvature constraint to bind (active cert) for the "
        "concave truth; got an empty active set"
    )

    def f(
        x_: "torch.Tensor",
        y_: "torch.Tensor",
        p_: "torch.Tensor",
        w_: "torch.Tensor",
    ) -> "torch.Tensor":
        return _scalar_objective(_forward(x_, y_, p_, w_))

    # Previously this raised NotImplementedError; it must now both run and
    # agree with finite differences to the float64 default tolerance.
    obj = f(x, y, penalty, weights)
    obj.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert y.grad is not None and torch.isfinite(y.grad).all()
    assert penalty.grad is not None and torch.isfinite(penalty.grad).all()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()

    assert torch.autograd.gradcheck(
        f,
        (x, y, penalty, weights),
        eps=_EPS,
        atol=_ATOL,
        rtol=_RTOL,
        nondet_tol=_NONDET_TOL,
    )
