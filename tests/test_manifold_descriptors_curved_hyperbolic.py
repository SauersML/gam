"""Constructibility + primitive-correctness tests for the curved / hyperbolic
manifold descriptors newly surfaced in :mod:`gamfit.manifolds`:
``Grassmann``, ``Stiefel``, ``Spd``, ``Poincare`` (the hyperbolic manifold).

These descriptors expose GEOMETRY PRIMITIVES ONLY (exp / log / metric /
distance / dimension), not fittable latent smooths. The tests here lock in:

* each descriptor is constructible and reports the correct intrinsic /
  ambient dimension (closed-form),
* the exp / log primitives round-trip (``log_p(exp_p(v)) == v``) on a tiny
  known point + tangent,
* Poincaré distance matches the textbook hyperbolic closed form on a tiny
  known case and is zero for identical points.

They require the compiled ``gamfit._rust`` extension; without it every
primitive raises :class:`gamfit.RustExtensionUnavailableError` and the module
is skipped.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gamfit import RustExtensionUnavailableError
from gamfit import manifolds


def _require_rust() -> None:
    from gamfit._binding import rust_module

    try:
        rust_module()
    except RustExtensionUnavailableError as exc:  # pragma: no cover - env specific
        pytest.skip(f"gamfit._rust unavailable: {exc}")


def setup_module(module):  # noqa: D401 - pytest hook
    _require_rust()


# --------------------------------------------------------------------------
# Dimension / constructibility (no Rust round-trip needed for the math, but
# `dimension` routes through Rust when available).
# --------------------------------------------------------------------------
def test_grassmann_constructible_and_dimensions():
    m = manifolds.Grassmann(k=2, n=5)
    assert m.dimension == 2 * (5 - 2)  # k*(n-k) = 6
    assert m.ambient_dim == 5 * 2  # n*k = 10
    assert "Grassmann" in repr(m)
    with pytest.raises(ValueError):
        manifolds.Grassmann(k=6, n=5)  # k > n rejected


def test_stiefel_constructible_and_dimensions():
    m = manifolds.Stiefel(k=2, n=4)
    assert m.dimension == 4 * 2 - 2 * (2 + 1) // 2  # nk - k(k+1)/2 = 5
    assert m.ambient_dim == 4 * 2  # n*k = 8
    assert "Stiefel" in repr(m)
    with pytest.raises(ValueError):
        manifolds.Stiefel(k=3, n=2)


def test_spd_constructible_and_dimensions():
    m = manifolds.Spd(n=3)
    assert m.dimension == 3 * (3 + 1) // 2  # n(n+1)/2 = 6
    assert m.ambient_dim == 3 * 3  # n*n = 9
    assert "Spd" in repr(m)
    with pytest.raises(ValueError):
        manifolds.Spd(n=0)


def test_poincare_constructible_and_dimensions():
    m = manifolds.Poincare(dim=3)
    assert m.dimension == 3
    assert m.ambient_dim == 3
    assert "Poincare" in repr(m)
    # Poincare IS the hyperbolic manifold: it defaults to (and requires)
    # negative curvature.
    assert float(m.curvature) < 0.0
    with pytest.raises(ValueError):
        manifolds.Poincare(dim=2, curvature=0.0)  # must be c < 0
    with pytest.raises(ValueError):
        manifolds.Poincare(dim=2, curvature=1.0)


# --------------------------------------------------------------------------
# Primitive correctness: exp/log round-trip on tiny known cases.
# --------------------------------------------------------------------------
def test_spd_exp_log_roundtrip_identity_base():
    m = manifolds.Spd(n=2)
    # Base point: 2x2 identity (SPD), flattened row-major.
    p = np.array([1.0, 0.0, 0.0, 1.0])
    # Symmetric tangent.
    v = np.array([0.1, 0.05, 0.05, -0.2])
    q = m.exp(p, v)
    v_back = m.log(p, q)
    assert np.allclose(v_back, v, atol=1e-8)


def test_stiefel_exp_log_roundtrip():
    m = manifolds.Stiefel(k=1, n=2)
    # A 1-frame in R^2: unit vector e1, flattened.
    p = np.array([1.0, 0.0])
    # Tangent in the horizontal space (orthogonal to p for Stiefel(1, n)=sphere).
    v = np.array([0.0, 0.3])
    q = m.exp(p, v)
    # Endpoint must stay a unit frame.
    assert math.isclose(float(np.linalg.norm(q)), 1.0, abs_tol=1e-8)
    v_back = m.log(p, q)
    assert np.allclose(v_back, v, atol=1e-7)


def test_grassmann_exp_log_roundtrip():
    m = manifolds.Grassmann(k=1, n=2)
    p = np.array([1.0, 0.0])  # span of e1, flattened n x k = 2 x 1
    v = np.array([0.0, 0.2])  # horizontal tangent
    q = m.exp(p, v)
    v_back = m.log(p, q)
    assert np.allclose(v_back, v, atol=1e-7)


def test_poincare_exp_log_roundtrip_at_origin():
    m = manifolds.Poincare(dim=2, curvature=-1.0)
    p = np.array([0.0, 0.0])
    v = np.array([0.4, -0.1])
    q = m.exp(p, v)
    # exp_0(v) = tanh(|v|) v / |v| (k=1) -> strictly inside the ball.
    assert float(np.linalg.norm(q)) < 1.0
    v_back = m.log(p, q)
    assert np.allclose(v_back, v, atol=1e-9)


def test_poincare_exp_log_roundtrip_off_origin():
    m = manifolds.Poincare(dim=2, curvature=-1.0)
    p = np.array([0.2, -0.1])
    v = np.array([0.15, 0.25])
    q = m.exp(p, v)
    assert float(np.linalg.norm(q)) < 1.0
    v_back = m.log(p, q)
    assert np.allclose(v_back, v, atol=1e-7)


def test_poincare_exp_origin_known_value():
    # exp_0([r, 0]) for c=-1 is [tanh(r), 0].
    m = manifolds.Poincare(dim=2, curvature=-1.0)
    q = m.exp(np.array([0.0, 0.0]), np.array([0.5, 0.0]))
    assert np.allclose(q, [math.tanh(0.5), 0.0], atol=1e-9)


def test_poincare_distance_known_case_and_self_zero():
    m = manifolds.Poincare(dim=2, curvature=-1.0)
    origin = np.array([0.0, 0.0])
    q = np.array([0.3, 0.0])
    # d_c(0, [r,0]) = (2/sqrt(k)) artanh(sqrt(k) r); k=1 -> 2 artanh(0.3).
    expected = 2.0 * math.atanh(0.3)
    assert math.isclose(float(m.distance(origin, q)), expected, rel_tol=1e-9)
    # Self-distance is exactly zero.
    assert float(m.distance(q, q)) == pytest.approx(0.0, abs=1e-12)


def test_poincare_metric_is_conformal_scaling_of_identity():
    m = manifolds.Poincare(dim=2, curvature=-1.0)
    p = np.array([0.3, 0.0])
    g = np.asarray(m.metric(p))
    lam = 2.0 / (1.0 + (-1.0) * float(np.dot(p, p)))  # conformal factor
    assert np.allclose(g, (lam * lam) * np.eye(2), atol=1e-12)
    # At the origin lambda_0 = 2, so g_0 = 4 I.
    g0 = np.asarray(m.metric(np.array([0.0, 0.0])))
    assert np.allclose(g0, 4.0 * np.eye(2), atol=1e-12)


def test_poincare_batch_exp_log_roundtrip():
    m = manifolds.Poincare(dim=2, curvature=-1.0)
    p = np.array([[0.0, 0.0], [0.1, 0.2]])
    v = np.array([[0.3, 0.0], [-0.1, 0.15]])
    q = m.exp(p, v)
    assert q.shape == (2, 2)
    v_back = m.log(p, q)
    assert np.allclose(v_back, v, atol=1e-7)


def test_poincare_project_keeps_interior_point_and_clamps_exterior():
    m = manifolds.Poincare(dim=2, curvature=-1.0)
    interior = np.array([0.3, 0.1])
    assert np.allclose(m.project(interior), interior, atol=1e-9)
    exterior = np.array([2.0, 0.0])  # outside the ball
    projected = np.asarray(m.project(exterior))
    assert float(np.linalg.norm(projected)) < 1.0
