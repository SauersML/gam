"""Exact value/gradient contract for the Rust-backed smooth threshold gate."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
torch = pytest.importorskip("torch")
pytest.importorskip("gamfit.torch")

from gamfit.torch.penalties import SmoothThresholdPenalty, _SmoothThresholdFn  # noqa: E402


def _rust_module() -> Any:
    binding = importlib.import_module("gamfit._binding")
    try:
        return binding.rust_module()
    except Exception as exc:
        pytest.skip(f"compiled gamfit._rust extension unavailable: {exc}")


def _gate(z: torch.Tensor, tau: torch.Tensor, eps: float) -> torch.Tensor:
    return _SmoothThresholdFn.apply(z, tau, float(eps))


def test_smooth_threshold_gate_gradcheck() -> None:
    torch.manual_seed(0)
    z = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    tau = (0.2 + torch.rand(6, dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda x, t: _gate(x, t, 0.1),
        (z, tau),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "z_np,tau_np,eps",
    [
        (
            np.array([[0.0, 0.4, 1.2], [0.8, 1.0, -0.5]], dtype=np.float64),
            np.array([0.4, 1.0, 0.2], dtype=np.float64),
            0.07,
        ),
        (
            np.array([[1e6, -1e6, 0.5], [0.5, 1.0, 2.0]], dtype=np.float64),
            np.array([0.5, 1.0, 2.0], dtype=np.float64),
            1e-3,
        ),
    ],
)
def test_smooth_threshold_gate_matches_rust(
    z_np: np.ndarray, tau_np: np.ndarray, eps: float
) -> None:
    rust = _rust_module()
    z = torch.tensor(z_np, dtype=torch.float64, requires_grad=True)
    tau = torch.tensor(tau_np, dtype=torch.float64, requires_grad=True)
    value = _gate(z, tau, eps)

    rust_value, rust_dz, rust_dtau = rust.smooth_threshold_gate_value_grad(
        np.ascontiguousarray(z_np), np.ascontiguousarray(tau_np), float(eps)
    )
    np.testing.assert_allclose(value.detach().numpy(), rust_value, atol=1e-12, rtol=0.0)

    grad_z, grad_tau = torch.autograd.grad(value, (z, tau), torch.ones_like(value))
    np.testing.assert_allclose(grad_z.detach().numpy(), rust_dz, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        grad_tau.detach().numpy(),
        np.asarray(rust_dtau).sum(axis=0),
        atol=1e-12,
        rtol=0.0,
    )


def test_smooth_threshold_penalty_gate_preserves_dtype_and_is_smooth() -> None:
    thresholds = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    penalty = SmoothThresholdPenalty(thresholds, smoothing_eps=0.1)
    z = torch.tensor(
        [[0.4, 1.2, 2.0], [0.6, 0.9, 1.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    out = penalty.gate(z)
    tau = thresholds.to(torch.float32).reshape(1, -1)
    expected = z * torch.sigmoid((z - tau) / 0.1)
    assert out.dtype == z.dtype
    assert out.device == z.device
    torch.testing.assert_close(out, expected)
    out.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
