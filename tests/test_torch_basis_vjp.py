"""gradcheck coverage for the analytic backward of gamfit.torch._basis primitives.

Each of the four forward primitives that now carry an exact analytic backward
is checked against torch's numerical Jacobian via ``torch.autograd.gradcheck``
(float64). The open B-spline derivative additionally passes ``gradgradcheck``
(second-order, input-location curvature). The sphere basis is checked for all
three kernels. The whole module is skipped when torch is unavailable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gamfit.torch._basis import (  # noqa: E402
    bspline_basis_derivative,
    duchon_basis,
    gaussian_weighted_ridge,
    gaussian_weighted_ridge_batch,
    sphere_basis,
)


# --------------------------------------------------------------------------- #
# Item A: bspline_basis_derivative
# --------------------------------------------------------------------------- #
def _bspline_knots(degree: int, n_interior: int) -> torch.Tensor:
    """Clamped open knot vector on [0, 1] with ``n_interior`` interior knots."""
    interior = torch.linspace(0.0, 1.0, n_interior + 2, dtype=torch.float64)[1:-1]
    left = torch.zeros(degree + 1, dtype=torch.float64)
    right = torch.ones(degree + 1, dtype=torch.float64)
    return torch.cat([left, interior, right])


@pytest.mark.parametrize("order", [1, 2])
def test_bspline_basis_derivative_gradcheck_open(order: int) -> None:
    degree = 3
    knots = _bspline_knots(degree, n_interior=3)
    t = torch.linspace(0.05, 0.95, 7, dtype=torch.float64).requires_grad_(True)

    def fn(tt: torch.Tensor) -> torch.Tensor:
        return bspline_basis_derivative(
            tt, knots, degree=degree, order=order, periodic=False
        )

    assert torch.autograd.gradcheck(fn, (t,), atol=1e-6, rtol=1e-4)


def test_bspline_basis_derivative_gradgradcheck_open() -> None:
    degree = 3
    knots = _bspline_knots(degree, n_interior=3)
    t = torch.linspace(0.05, 0.95, 6, dtype=torch.float64).requires_grad_(True)

    def fn(tt: torch.Tensor) -> torch.Tensor:
        return bspline_basis_derivative(tt, knots, degree=degree, order=1, periodic=False)

    assert torch.autograd.gradgradcheck(fn, (t,), atol=1e-5, rtol=1e-3)


# --------------------------------------------------------------------------- #
# Item C: sphere_basis (grad through points)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kernel", ["sobolev", "pseudo", "harmonic"])
def test_sphere_basis_gradcheck_points(kernel: str) -> None:
    n_centers = 4 if kernel != "harmonic" else 3
    # (lat, lon) in degrees, kept well inside the valid range.
    points = torch.tensor(
        [[12.0, 30.0], [-20.0, 100.0], [40.0, -60.0], [5.0, 170.0]],
        dtype=torch.float64,
    ).requires_grad_(True)

    def fn(pts: torch.Tensor) -> torch.Tensor:
        design, _penalty = sphere_basis(
            pts, n_centers, penalty_order=2, kernel=kernel, radians=False
        )
        return design

    assert torch.autograd.gradcheck(fn, (points,), atol=1e-6, rtol=1e-4)


# --------------------------------------------------------------------------- #
# Item D: duchon_basis (grad through points)
#
# The forward design and the analytic jets must come from ONE Rust builder so
# the input gradient is the exact derivative of the returned forward (gam#2097);
# the basis-only forward applies a batch-global data-metric reparam the jets do
# not, which used to make the VJP wrong / un-broadcastable.
# --------------------------------------------------------------------------- #
_DUCHON_POINTS = torch.tensor(
    [[0.15, 0.25], [0.35, 0.65], [0.55, 0.45], [0.75, 0.85], [0.25, 0.55]],
    dtype=torch.float64,
)
_DUCHON_CENTERS = torch.tensor(
    [[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9], [0.5, 0.5], [0.3, 0.7]],
    dtype=torch.float64,
)


@pytest.mark.parametrize("m", [2, 3])
def test_duchon_basis_gradcheck_points(m: int) -> None:
    points = _DUCHON_POINTS.clone().requires_grad_(True)

    def fn(pts: torch.Tensor) -> torch.Tensor:
        return duchon_basis(pts, centers=_DUCHON_CENTERS, m=m)

    assert torch.autograd.gradcheck(fn, (points,), atol=1e-6, rtol=1e-4)


def test_duchon_basis_gradgradcheck_points() -> None:
    # Second-order (input-location Hessian) routes through the Rust second jet.
    points = _DUCHON_POINTS[:4].clone().requires_grad_(True)

    def fn(pts: torch.Tensor) -> torch.Tensor:
        return duchon_basis(pts, centers=_DUCHON_CENTERS, m=2)

    assert torch.autograd.gradgradcheck(fn, (points,), atol=1e-5, rtol=1e-3)


def test_duchon_basis_gradcheck_width_reducing_config() -> None:
    # K=10 centers in 2D: the basis-only forward's data-metric reparam drops a
    # near-null kernel mode (9 columns) while the jet builder keeps all 10, so
    # the two-builder path raised a shape-mismatch RuntimeError here. Sharing one
    # builder keeps the width consistent (10) and the gradient exact.
    g = torch.Generator().manual_seed(0)
    centers = torch.rand(10, 2, generator=g, dtype=torch.float64)
    points = torch.rand(4, 2, generator=g, dtype=torch.float64).requires_grad_(True)

    out = duchon_basis(points, centers=centers, m=2)
    assert out.shape == (4, 10)
    assert torch.autograd.gradcheck(
        lambda pts: duchon_basis(pts, centers=centers, m=2),
        (points,),
        atol=1e-6,
        rtol=1e-4,
    )


def test_duchon_basis_gradcheck_1d_and_periodic() -> None:
    # 1D open smooth.
    p1 = torch.tensor(
        [[0.2], [0.4], [0.6], [0.8]], dtype=torch.float64
    ).requires_grad_(True)
    c1 = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda x: duchon_basis(x, centers=c1, m=2), (p1,), atol=1e-6, rtol=1e-4
    )
    # 1D circle (mixed-periodicity builder): the forward and jets share the
    # additive-kernel design too.
    pc = torch.tensor(
        [[0.2], [1.1], [2.4], [3.0]], dtype=torch.float64
    ).requires_grad_(True)
    cc = torch.tensor(
        [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float64
    )
    assert torch.autograd.gradcheck(
        lambda x: duchon_basis(x, centers=cc, m=2, periodic_per_axis=(True,)),
        (pc,),
        atol=1e-5,
        rtol=1e-3,
    )


# --------------------------------------------------------------------------- #
# Item B: gaussian_weighted_ridge (and _batch)
# --------------------------------------------------------------------------- #
def _ridge_inputs(n: int, m: int, d: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, m, generator=g, dtype=torch.float64)
    Y = torch.randn(n, d, generator=g, dtype=torch.float64)
    # SPD penalty (structural) and strictly positive weights.
    P = torch.randn(m, m, generator=g, dtype=torch.float64)
    penalty = P @ P.transpose(0, 1) + m * torch.eye(m, dtype=torch.float64)
    weights = torch.rand(n, generator=g, dtype=torch.float64) + 0.5
    return X, Y, penalty, weights


def test_gaussian_weighted_ridge_gradcheck() -> None:
    n, m, d = 6, 3, 2
    X, Y, penalty, weights = _ridge_inputs(n, m, d)
    X = X.requires_grad_(True)
    Y = Y.requires_grad_(True)
    penalty = penalty.requires_grad_(True)
    weights = weights.requires_grad_(True)
    ridge_lambda = 0.7

    def fn(Xv, Yv, Pv, Wv):
        coef, fitted = gaussian_weighted_ridge(
            Xv, Yv, Pv, Wv, ridge_lambda=ridge_lambda
        )
        return coef, fitted

    assert torch.autograd.gradcheck(
        fn, (X, Y, penalty, weights), atol=1e-6, rtol=1e-4
    )


def test_gaussian_weighted_ridge_batch_gradcheck_no_padding() -> None:
    k, n, m, d = 2, 5, 3, 2
    Xs, Ys, ws = [], [], []
    for s in range(k):
        X, Y, penalty, weights = _ridge_inputs(n, m, d, seed=s + 1)
        Xs.append(X)
        Ys.append(Y)
        ws.append(weights)
    X = torch.stack(Xs).requires_grad_(True)
    Y = torch.stack(Ys).requires_grad_(True)
    # Shared SPD penalty.
    _, _, penalty, _ = _ridge_inputs(n, m, d, seed=99)
    penalty = penalty.requires_grad_(True)
    weights = torch.stack(ws).requires_grad_(True)
    ridge_lambda = 0.5

    def fn(Xv, Yv, Pv, Wv):
        coef, fitted = gaussian_weighted_ridge_batch(
            Xv, Yv, Pv, Wv, ridge_lambda=ridge_lambda
        )
        return coef, fitted

    assert torch.autograd.gradcheck(
        fn, (X, Y, penalty, weights), atol=1e-6, rtol=1e-4
    )


def test_gaussian_weighted_ridge_batch_gradcheck_padded() -> None:
    k, n_max, m, d = 2, 6, 3, 2
    row_counts = torch.tensor([6, 4], dtype=torch.int64)
    Xs, Ys, ws = [], [], []
    for s in range(k):
        X, Y, _penalty, weights = _ridge_inputs(n_max, m, d, seed=s + 7)
        Xs.append(X)
        Ys.append(Y)
        ws.append(weights)
    X = torch.stack(Xs).requires_grad_(True)
    Y = torch.stack(Ys).requires_grad_(True)
    _, _, penalty, _ = _ridge_inputs(n_max, m, d, seed=123)
    penalty = penalty.requires_grad_(True)
    weights = torch.stack(ws).requires_grad_(True)
    ridge_lambda = 0.9

    def fn(Xv, Yv, Pv, Wv):
        coef, fitted = gaussian_weighted_ridge_batch(
            Xv, Yv, Pv, Wv, ridge_lambda=ridge_lambda, row_counts=row_counts
        )
        # Padded fitted rows are not meaningful outputs; exclude them so the
        # numerical/analytic Jacobians compare only the active prefix.
        masked = []
        for kk in range(Xv.shape[0]):
            masked.append(fitted[kk, : int(row_counts[kk])])
        return coef, torch.cat(masked, dim=0)

    assert torch.autograd.gradcheck(
        fn, (X, Y, penalty, weights), atol=1e-6, rtol=1e-4
    )
