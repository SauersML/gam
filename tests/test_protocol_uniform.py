"""Uniform callable-descriptor protocol tests.

Covers the three sibling protocols introduced in :mod:`gamfit._protocol`:

* :class:`ManifoldDescriptor` — ``exp``, ``log``, ``metric``, ``geodesic``,
  ``dimension`` returning torch tensors with grad through inputs.
* :class:`BasisDescriptor` — ``evaluate`` (and ``__call__``) returning a
  design matrix.
* :class:`PenaltyDescriptor` — composition via ``+`` produces a sum
  composite whose ``value / value_grad / hvp / hessian_diag`` are sums
  of the parts.

The :class:`Smooth` compose API (``Smooth(latent=..., basis=...,
penalty=...)``) is also exercised end-to-end, including the eager
dim-mismatch check.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gamfit
from gamfit.manifolds import Circle as ManifoldCircle
from gamfit.manifolds import Sphere as ManifoldSphere


def test_circle_exp_returns_tensor_with_grad_through_v() -> None:
    M = ManifoldCircle()
    assert M.dimension == 1
    # Circle is parameterized by a single angle coordinate, matching the Rust
    # ``gam::geometry::Circle`` (ambient == intrinsic == 1). There is no R^2
    # unit-vector embedding; see issue #397.
    assert M.ambient_dim == 1
    # Point and tangent are both 1-D angle coordinates; exp wraps to (-pi, pi].
    p = torch.tensor([0.5], dtype=torch.float64)
    v = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
    q = M.exp(p, v)
    assert isinstance(q, torch.Tensor)
    assert q.dtype == torch.float64
    assert q.shape == (1,)
    # exp_theta(v) = wrap(theta + v); for small angles this is just theta + v.
    assert torch.allclose(q.detach(), torch.tensor([0.6], dtype=torch.float64), atol=1e-12)
    # Gradient flows back through v (the flat-circle VJP is the identity).
    q.sum().backward()
    assert v.grad is not None
    assert torch.isfinite(v.grad).all()
    assert torch.allclose(v.grad, torch.tensor([1.0], dtype=torch.float64), atol=1e-12)


def test_circle_cylinder_ambient_dim_matches_rust_source_of_truth() -> None:
    """Regression for issue #397: the Python descriptor's documented contract
    and its cached ``_ambient_dim`` must agree with the Rust manifold, which
    uses the 1-D angle parameterization (ambient == intrinsic). Following the
    old ``R^2`` "unit 2-vector" docstring raised a hard ``GamError``."""
    from gamfit.manifolds import CylinderManifold

    circle = ManifoldCircle()
    # Rust-backed property and the cached fallback must agree, and both == 1.
    assert circle.ambient_dim == 1
    assert circle._ambient_dim == 1
    assert circle.ambient_dim == circle._ambient_dim
    # The numpy path exercises the Rust manifold directly: the 1-D angle form
    # must succeed and wrap correctly.
    out = circle.exp(np.array([0.5]), np.array([0.1]))
    assert out.shape == (1,)
    assert math.isclose(float(out[0]), 0.6, abs_tol=1e-12)

    for open_dim in (0, 1, 3):
        cyl = CylinderManifold(open_dim)
        assert cyl.ambient_dim == 1 + open_dim
        assert cyl._ambient_dim == 1 + open_dim
        assert cyl.ambient_dim == cyl._ambient_dim
        # A point is [theta, x_1, ..., x_k] of width 1 + open_dim.
        p = np.zeros(1 + open_dim, dtype=np.float64)
        v = np.zeros(1 + open_dim, dtype=np.float64)
        p[0] = 0.5
        v[0] = 0.1
        q = cyl.exp(p, v)
        assert q.shape == (1 + open_dim,)
        assert math.isclose(float(q[0]), 0.6, abs_tol=1e-12)


def test_sphere_metric_is_symmetric_psd() -> None:
    M = ManifoldSphere(intrinsic_dim=2)
    for _ in range(5):
        p_raw = torch.randn(3, dtype=torch.float64)
        p = p_raw / torch.linalg.vector_norm(p_raw)
        g = M.metric(p)
        # Symmetric
        assert torch.allclose(g, g.T, atol=1e-10)
        # PSD on ambient R^3 (the Rust metric tensor returns the induced
        # metric on the tangent space, padded with a zero row/column for
        # the normal direction).
        eigvals = torch.linalg.eigvalsh(g)
        assert (eigvals >= -1e-10).all()


def test_fourier_evaluate_matches_periodic_harmonic() -> None:
    fourier = gamfit.Fourier(harmonics=3)
    ph = gamfit.PeriodicHarmonic(num_basis=7)
    theta = torch.linspace(0.0, 2.0 * math.pi, 32, dtype=torch.float64)
    phi_a = fourier.evaluate(theta)
    phi_b = ph.evaluate(theta)
    assert phi_a.shape == (32, 7)
    assert torch.allclose(phi_a, phi_b, atol=1e-14)
    # Callable surface
    assert torch.allclose(fourier(theta), phi_a, atol=1e-14)


def test_penalty_composition_hvp_is_sum_of_parts() -> None:
    pa = gamfit.ARDPenalty(weight=0.1)
    pb = gamfit.OrderedBetaBernoulliPenalty(alpha=1.0, tau=1.0)
    composite = pa + pb
    from gamfit._composite_penalty import CompositePenalty
    assert isinstance(composite, CompositePenalty)
    assert len(composite) == 2

    t = torch.randn(4, 3, dtype=torch.float64) * 0.3
    v = torch.randn_like(t)
    hv_a = pa.hvp(t, v)
    hv_b = pb.hvp(t, v)
    hv_c = composite.hvp(t, v)
    assert torch.allclose(hv_c, hv_a + hv_b, atol=1e-10)


def test_smooth_compose_circle_fourier_evaluate_matches_basis() -> None:
    sm = gamfit.Smooth(
        latent=ManifoldCircle(),
        basis=gamfit.Fourier(harmonics=3),
        penalty=gamfit.ARDPenalty(weight=0.1),
    )
    theta = torch.linspace(0.0, 2.0 * math.pi, 16, dtype=torch.float64)
    phi_sm = sm.evaluate(theta)
    phi_b = sm.basis.evaluate(theta)
    assert torch.allclose(phi_sm, phi_b, atol=1e-14)
    assert sm(theta).shape == phi_sm.shape


def test_basis_jacobian_evaluates_descriptor_once() -> None:
    class CountingBasis(gamfit.BasisDescriptor):
        def __init__(self) -> None:
            self.evaluate_calls = 0

        def evaluate(self, t: Any) -> Any:
            self.evaluate_calls += 1
            return t**2

    basis = CountingBasis()
    t = torch.tensor([1.0, 2.0], dtype=torch.float64)

    jacobian = basis.jacobian(t)

    assert basis.evaluate_calls == 1
    assert torch.allclose(jacobian, torch.diag(2.0 * t))


def test_smooth_dim_mismatch_raises_eagerly() -> None:
    # Sphere has dimension=2; Fourier(harmonics=3) has input_dim=1.
    with pytest.raises(ValueError, match="incompatible|input_dim|dimension"):
        gamfit.Smooth(
            latent=ManifoldSphere(intrinsic_dim=2),
            basis=gamfit.Fourier(harmonics=3),
        )


def test_smooth_to_dict_roundtrip() -> None:
    sm = gamfit.Smooth(
        latent=ManifoldCircle(),
        basis=gamfit.Fourier(harmonics=3),
        name="phase",
    )
    d = sm.to_dict()
    assert d["kind"] == "composed_smooth"
    assert d["name"] == "phase"
    assert d["latent"]["kind"] == "circle"
    assert d["basis"]["kind"] == "periodic_harmonic"

    sm2 = gamfit.Smooth.from_dict(d)
    theta = torch.linspace(0.0, 2.0 * math.pi, 8, dtype=torch.float64)
    assert torch.allclose(sm.evaluate(theta), sm2.evaluate(theta), atol=1e-14)
