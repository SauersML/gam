"""Cross-frame dispatch contract for gamfit primitives.

The same Rust kernel must drive every numerical frame the user might
already be working in (NumPy / Torch / JAX). The frame is auto-detected
from input types; the return is in the user's native frame; analytic
gradients route through frame-native autograd.

Torch and JAX are optional dependencies — every test that needs one
skips cleanly when the framework is missing.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit
from gamfit._frame import Frame, detect_frame


# ---------------------------------------------------------------------------
# Optional framework probes
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent
    import torch as _torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _torch = None
    _HAS_TORCH = False

try:  # pragma: no cover - environment-dependent
    import jax as _jax
    import jax.numpy as _jnp
    _HAS_JAX = True
except ImportError:  # pragma: no cover
    _jax = None
    _jnp = None
    _HAS_JAX = False


needs_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
needs_jax = pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")


# ---------------------------------------------------------------------------
# Frame detection
# ---------------------------------------------------------------------------


def test_detect_frame_defaults_to_numpy() -> None:
    assert detect_frame() is Frame.NUMPY
    assert detect_frame(np.zeros(3)) is Frame.NUMPY
    assert detect_frame(np.zeros(3), np.ones(2)) is Frame.NUMPY


@needs_torch
def test_detect_frame_torch() -> None:
    t = _torch.zeros(3)
    assert detect_frame(t) is Frame.TORCH
    # numpy alongside torch does not force a switch
    assert detect_frame(t, np.zeros(2)) is Frame.TORCH


@needs_jax
def test_detect_frame_jax() -> None:
    a = _jnp.zeros(3)
    assert detect_frame(a) is Frame.JAX
    assert detect_frame(a, np.zeros(2)) is Frame.JAX


@needs_torch
@needs_jax
def test_detect_frame_mixed_raises() -> None:
    t = _torch.zeros(3)
    a = _jnp.zeros(3)
    with pytest.raises(TypeError, match="same frame"):
        detect_frame(t, a)


# ---------------------------------------------------------------------------
# Cylinder descriptor
# ---------------------------------------------------------------------------


def _cylinder_inputs(n: int = 13) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ell = np.linspace(0.0, 1.0, n)
    return theta, ell


def test_cylinder_numpy_frame() -> None:
    theta, ell = _cylinder_inputs()
    phi = gamfit.Cylinder(n_knots=(7, 4)).evaluate(theta, ell)
    assert isinstance(phi, np.ndarray)
    assert phi.ndim == 2
    assert phi.shape[0] == theta.size
    assert np.all(np.isfinite(phi))


@needs_torch
def test_cylinder_torch_matches_numpy() -> None:
    theta, ell = _cylinder_inputs()
    phi_np = gamfit.Cylinder(n_knots=(7, 4)).evaluate(theta, ell)
    phi_t = gamfit.Cylinder(n_knots=(7, 4)).evaluate(
        _torch.as_tensor(theta), _torch.as_tensor(ell)
    )
    assert isinstance(phi_t, _torch.Tensor)
    np.testing.assert_allclose(phi_t.detach().cpu().numpy(), phi_np, atol=1e-10)


@needs_jax
def test_cylinder_jax_matches_numpy() -> None:
    theta, ell = _cylinder_inputs()
    phi_np = gamfit.Cylinder(n_knots=(7, 4)).evaluate(theta, ell)
    phi_j = gamfit.Cylinder(n_knots=(7, 4)).evaluate(
        _jnp.asarray(theta), _jnp.asarray(ell)
    )
    # jax array module starts with "jax" or "jaxlib"
    assert type(phi_j).__module__.startswith(("jax", "jaxlib"))
    np.testing.assert_allclose(np.asarray(phi_j), phi_np, atol=1e-10)


# ---------------------------------------------------------------------------
# Sphere descriptor
# ---------------------------------------------------------------------------


def _sphere_inputs() -> np.ndarray:
    rng = np.random.default_rng(0)
    lat = rng.uniform(-60.0, 60.0, size=6)
    lon = rng.uniform(-150.0, 150.0, size=6)
    return np.column_stack([lat, lon])


def test_sphere_numpy_frame() -> None:
    pts = _sphere_inputs()
    sph = gamfit.Sphere(n_centers=12)
    phi = sph.evaluate(pts[:, 0], pts[:, 1])
    assert isinstance(phi, np.ndarray)
    assert phi.shape[0] == pts.shape[0]


# ---------------------------------------------------------------------------
# Penalty cross-frame value_grad
# ---------------------------------------------------------------------------


def _ard_target(rng_seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return rng.normal(size=(4, 3))


def test_ard_penalty_value_grad_numpy() -> None:
    t = _ard_target()
    pen = gamfit.ARDPenalty(weight=1.0)
    value, grad = pen.value_grad(t)
    assert isinstance(grad, np.ndarray)
    assert grad.shape == t.shape
    assert np.isfinite(value)
    assert np.all(np.isfinite(grad))


@needs_torch
def test_ard_penalty_torch_grad_matches_numpy() -> None:
    t_np = _ard_target()
    pen = gamfit.ARDPenalty(weight=1.0)
    value_np, grad_np = pen.value_grad(t_np)

    t_t = _torch.tensor(t_np, dtype=_torch.float64, requires_grad=True)
    value_t, grad_t = pen.value_grad(t_t)
    assert isinstance(value_t, _torch.Tensor)
    np.testing.assert_allclose(float(value_t.detach()), value_np, atol=1e-10)
    np.testing.assert_allclose(grad_t.detach().cpu().numpy(), grad_np, atol=1e-10)

    # autograd through the value tensor must match the analytic gradient
    (grad_auto,) = _torch.autograd.grad(value_t, t_t)
    np.testing.assert_allclose(grad_auto.detach().cpu().numpy(), grad_np, atol=1e-8)


@needs_jax
def test_ard_penalty_jax_grad_matches_numpy() -> None:
    t_np = _ard_target()
    pen = gamfit.ARDPenalty(weight=1.0)
    value_np, grad_np = pen.value_grad(t_np)

    t_j = _jnp.asarray(t_np)
    value_j, grad_j = pen.value_grad(t_j)
    np.testing.assert_allclose(float(np.asarray(value_j)), value_np, atol=1e-10)
    np.testing.assert_allclose(np.asarray(grad_j), grad_np, atol=1e-10)

    # jax.grad of the value-only callable must match the analytic gradient
    def _value_only(x: _jnp.ndarray) -> _jnp.ndarray:
        v, _ = pen.value_grad(x)
        return v

    grad_jax = _jax.grad(_value_only)(t_j)
    np.testing.assert_allclose(np.asarray(grad_jax), grad_np, atol=1e-8)


# ---------------------------------------------------------------------------
# Mixed-frame rejection
# ---------------------------------------------------------------------------


@needs_torch
def test_mixed_frame_inputs_raise_typeerror() -> None:
    theta_np = np.linspace(0.0, 1.0, 5)
    ell_t = _torch.linspace(0.0, 1.0, 5)
    with pytest.raises(TypeError, match="same frame"):
        gamfit.Cylinder(n_knots=(5, 4)).evaluate(theta_np, ell_t)
