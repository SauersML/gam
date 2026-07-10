"""Value/gradient contract for the Rust-backed JumpReLU activation-gate STE.

The torch JumpReLU gate (``gamfit.torch.penalties.JumpReLUPenalty.gate`` via
``_JumpReLUSTEFn``) calls the Rust source of truth
``gam::terms::analytic_penalties::jumprelu_gate_value_grad``. This test pins the
Torch bridge's forward value and both straight-through gradients (logit/``z``
and threshold/``τ``) to that kernel, including the exact threshold boundary,
all-zero rows, and extreme magnitudes.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")

from gamfit.torch.penalties import JumpReLUPenalty, _JumpReLUSTEFn  # noqa: E402


def _rust_module() -> Any:
    """Return the compiled ``gamfit._rust`` extension, or skip if unavailable."""
    binding = importlib.import_module("gamfit._binding")
    try:
        return binding.rust_module()
    except Exception as exc:  # RustExtensionUnavailableError on pure-Python shims
        pytest.skip(f"compiled gamfit._rust extension unavailable: {exc}")


def _gate(z: torch.Tensor, tau: torch.Tensor, smoothing_eps: float) -> torch.Tensor:
    apply = _JumpReLUSTEFn.apply
    return apply(z, tau, float(smoothing_eps))  # type: ignore[no-any-return]


def test_jumprelu_ste_gradcheck() -> None:
    """The torch autograd backward is consistent with its own forward.

    Validates the straight-through ``z`` gradient and accumulated threshold
    gradient against finite differences of the Rust-backed forward. ``gradcheck``
    perturbs coordinates by ``eps`` so we
    keep the samples away from the exact ``z == τ`` jump where the hard gate is
    discontinuous.
    """
    torch.manual_seed(0)
    rows, cols = 4, 6
    smoothing_eps = 1e-2
    # Offset logits well clear of the thresholds so the FD probe never straddles
    # the discontinuity of the hard gate (the STE surrogate is smooth; the
    # returned value is not).
    thresholds = torch.rand(cols, dtype=torch.float64, requires_grad=True)
    logits = (
        thresholds.detach() + 1.0 + torch.rand(rows, cols, dtype=torch.float64)
    ).requires_grad_(True)

    assert torch.autograd.gradcheck(
        lambda x, t: _gate(x, t, smoothing_eps),
        (logits, thresholds),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
    )


def _assert_matches_rust(
    logits_np: np.ndarray, tau_np: np.ndarray, smoothing_eps: float, atol: float
) -> None:
    rust = _rust_module()
    rows, cols = logits_np.shape

    logits = torch.tensor(logits_np, dtype=torch.float64, requires_grad=True)
    tau = torch.tensor(tau_np, dtype=torch.float64, requires_grad=True)

    value = _gate(logits, tau, smoothing_eps)

    # Reference: the Rust source of truth over the whole (N, F) matrix.
    rust_value, rust_dphi_dz, rust_dphi_dtau = rust.jumprelu_gate_value_grad(
        np.ascontiguousarray(logits_np),
        np.ascontiguousarray(tau_np),
        float(smoothing_eps),
    )
    rust_value = np.asarray(rust_value, dtype=np.float64)
    rust_dphi_dz = np.asarray(rust_dphi_dz, dtype=np.float64)
    rust_dphi_dtau = np.asarray(rust_dphi_dtau, dtype=np.float64)

    np.testing.assert_allclose(
        value.detach().cpu().numpy(),
        rust_value,
        rtol=0.0,
        atol=atol,
        err_msg="torch JumpReLU gate forward value must match the Rust kernel.",
    )

    # Seed the upstream with ones so each output cell contributes its own
    # ``∂φ̃/∂z`` — exactly the Rust ``dphi_dz`` matrix — and the threshold grad
    # is the per-column sum of ``∂φ̃/∂τ = −slope``.
    upstream = torch.ones_like(value)
    (z_grad, tau_grad) = torch.autograd.grad(value, (logits, tau), upstream)

    np.testing.assert_allclose(
        z_grad.detach().cpu().numpy(),
        rust_dphi_dz,
        rtol=0.0,
        atol=atol,
        err_msg="torch JumpReLU z gradient must match the Rust STE derivative.",
    )
    np.testing.assert_allclose(
        tau_grad.detach().cpu().numpy(),
        rust_dphi_dtau.sum(axis=0),
        rtol=0.0,
        atol=atol,
        err_msg="torch JumpReLU threshold gradient must be the column-sum of the Rust ∂φ̃/∂τ.",
    )


def test_jumprelu_ste_matches_rust_generic() -> None:
    """Torch forward value + both STE gradients match the Rust kernel to 1e-12."""
    rng = np.random.default_rng(1234)
    rows, cols = 5, 7
    logits_np = rng.standard_normal((rows, cols))
    tau_np = np.abs(rng.standard_normal(cols)) + 0.1
    _assert_matches_rust(logits_np, tau_np, smoothing_eps=1e-3, atol=1e-12)


def test_jumprelu_ste_matches_rust_threshold_boundary() -> None:
    """Rows sitting exactly on and infinitesimally around ``z == τ``.

    The hard gate ``1[z > τ]`` is strict, so ``z == τ`` must return 0 while the
    smooth STE gradient stays finite (``g = σ(0) = 1/2``). Parity here proves the
    ``torch.where(z > τ, ...)`` strict comparison matches the Rust ``z > tau``.
    """
    cols = 5
    tau_np = np.linspace(0.2, 1.5, cols)
    tiny = 1e-9
    logits_np = np.stack(
        [
            tau_np,  # exactly on the boundary -> value 0
            tau_np + tiny,  # just above -> value = z
            tau_np - tiny,  # just below -> value 0
            np.zeros(cols),  # zero row, all below positive thresholds
        ]
    )
    _assert_matches_rust(logits_np, tau_np, smoothing_eps=1e-3, atol=1e-12)


def test_jumprelu_ste_matches_rust_extreme_values() -> None:
    """Extreme magnitudes saturate the sigmoid gate to 0/1 without overflow.

    Large positive/negative ``(z − τ)/ε`` drives ``g`` to the sigmoid limits; the
    ``z·g·(1−g)`` slope must underflow to 0 identically in both paths (the STE
    signal vanishes far from the jump), and the stable logistic must not overflow.
    """
    cols = 4
    tau_np = np.array([0.5, 1.0, 2.0, 5.0])
    logits_np = np.array(
        [
            [1e6, -1e6, 1e3, -1e3],
            [0.5, 1.0, 2.0, 5.0],  # exactly on thresholds
            [-1e12, 1e12, 0.0, 100.0],
        ]
    )
    _assert_matches_rust(logits_np, tau_np, smoothing_eps=1e-3, atol=1e-12)


def test_jumprelu_penalty_gate_uses_ste_and_preserves_dtype_device() -> None:
    """``JumpReLUPenalty.gate`` returns the hard gate on the input dtype/device.

    No CPU/float64 round-trip: a float32 input stays float32 and the forward is
    exactly ``z · 1[z > τ]``.
    """
    thr = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    pen = JumpReLUPenalty(thr, weight=1.0, smoothing_eps=1e-3)
    z = torch.tensor(
        [[0.4, 1.2, 2.0], [0.6, 0.9, 1.5]], dtype=torch.float32, requires_grad=True
    )
    out = pen.gate(z)
    assert out.dtype == torch.float32
    assert out.device == z.device
    expected = torch.where(
        z > thr.to(torch.float32).reshape(1, -1), z, torch.zeros_like(z)
    )
    assert torch.equal(out.detach(), expected.detach())
    # Backward flows (STE keeps a live gradient even for gated-off coordinates).
    out.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
