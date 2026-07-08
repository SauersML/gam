"""Parity gate for the device-resident periodic basis lane (gam_sae::basis_gpu).

The torch manifold-SAE bridge (`_BasisWithJetFn`) gained a CUDA fast path: the
Rust NVRTC kernel writes phi/jet into torch-owned device buffers with no host
round-trip. The CPU `basis_with_jet("periodic", ...)` evaluator is the single
source of the basis math, so the device lane must reproduce it to rounding —
these tests pin the two together (values AND the jet the backward consumes).

Skipped where no CUDA device exists (the lane itself is unreachable there: the
bridge falls back to the CPU path).
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gamfit._binding import rust_module

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required for the device basis lane"
)


@requires_cuda
def test_periodic_basis_cuda_matches_cpu_oracle():
    n, n_harm = 4097, 5
    m = 2 * n_harm + 1
    rng = np.random.default_rng(0)
    # Deliberately outside [0, 1): the evaluator is periodic in t, not clamped,
    # and the encoder feeds unconstrained phases through the sigmoid upstream.
    t_np = rng.uniform(-2.0, 2.0, size=(n, 1))
    phi_cpu, jet_cpu, _penalty = rust_module().basis_with_jet(
        "periodic", t_np, {"n_harmonics": n_harm}
    )
    t = torch.as_tensor(t_np, dtype=torch.float64, device="cuda").contiguous()
    phi = torch.empty((n, m), dtype=torch.float64, device="cuda")
    jet = torch.empty((n, m, 1), dtype=torch.float64, device="cuda")
    torch.cuda.synchronize()
    rust_module().sae_periodic_basis_with_jet_cuda(
        int(t.device.index or 0), t.data_ptr(), n, n_harm, phi.data_ptr(), jet.data_ptr()
    )
    np.testing.assert_allclose(phi.cpu().numpy(), np.asarray(phi_cpu), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        jet.cpu().numpy().reshape(n, m),
        np.asarray(jet_cpu).reshape(n, m),
        rtol=0.0,
        atol=1e-10,  # jet carries the 2*pi*h frequency factor (~31 at h=5)
    )


@requires_cuda
def test_duchon_basis_cuda_matches_cpu_oracle():
    n, dim, n_centers, m_order = 2049, 2, 12, 2
    rng = np.random.default_rng(1)
    t_np = rng.uniform(-1.5, 1.5, size=(n, dim))
    centers = rng.uniform(-1.0, 1.0, size=(n_centers, dim))
    phi_cpu, jet_cpu, _penalty = rust_module().basis_with_jet(
        "duchon", t_np, {"centers": centers, "m": m_order}
    )
    phi_cpu = np.asarray(phi_cpu)
    jet_cpu = np.asarray(jet_cpu)
    width = rust_module().sae_duchon_device_basis_width(0, centers, m_order)
    assert width == phi_cpu.shape[1], (width, phi_cpu.shape)
    t = torch.as_tensor(t_np, dtype=torch.float64, device="cuda").contiguous()
    phi = torch.empty((n, width), dtype=torch.float64, device="cuda")
    jet = torch.empty((n, width, dim), dtype=torch.float64, device="cuda")
    torch.cuda.synchronize()
    out_width = rust_module().sae_duchon_basis_with_jet_cuda(
        0, (t.data_ptr(), phi.data_ptr(), jet.data_ptr()), n, centers, m_order
    )
    assert out_width == width
    np.testing.assert_allclose(phi.cpu().numpy(), phi_cpu, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(jet.cpu().numpy(), jet_cpu, rtol=1e-10, atol=1e-11)


@requires_cuda
def test_basis_bridge_autograd_device_lane_matches_cpu_lane():
    from gamfit.torch.manifold_sae import _BasisWithJetFn

    n, n_harm = 257, 3
    params = json.dumps({"n_harmonics": n_harm})
    t_cpu = torch.rand((n, 1), dtype=torch.float64, requires_grad=True)
    t_gpu = t_cpu.detach().clone().to("cuda").requires_grad_(True)
    phi_cpu = _BasisWithJetFn.apply(t_cpu, "periodic", params)
    phi_gpu = _BasisWithJetFn.apply(t_gpu, "periodic", params)
    torch.testing.assert_close(phi_gpu.cpu(), phi_cpu, rtol=0.0, atol=1e-12)
    weights = torch.randn_like(phi_cpu)
    (phi_cpu * weights).sum().backward()
    (phi_gpu * weights.to("cuda")).sum().backward()
    assert t_cpu.grad is not None and t_gpu.grad is not None
    torch.testing.assert_close(t_gpu.grad.cpu(), t_cpu.grad, rtol=0.0, atol=1e-10)
