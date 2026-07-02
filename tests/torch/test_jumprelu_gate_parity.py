"""Bit-parity contract for the pure-torch bounded JumpReLU gate.

The torch bounded JumpReLU gate (``gamfit.torch.penalties.jumprelu_bounded_gate``)
is a pure-vectorized, on-device transcription of the Rust source of truth
``gam_sae::assignment::jumprelu_row_value_grad`` (crates/gam-sae/src/
assignment.rs:1079). Rust stays the single source of truth: this test pins the
torch forward value AND both gradients (logit grad, threshold grad) to the Rust
kernel row-by-row to double-precision tolerance when the compiled extension is
importable, and always exercises the torch autograd backward via ``gradcheck``.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")

from gamfit.torch.penalties import jumprelu_bounded_gate  # noqa: E402


def _rust_module() -> Any:
    """Return the compiled ``gamfit._rust`` extension, or skip if unavailable."""
    binding = importlib.import_module("gamfit._binding")
    try:
        return binding.rust_module()
    except Exception as exc:  # RustExtensionUnavailableError on pure-Python shims
        pytest.skip(f"compiled gamfit._rust extension unavailable: {exc}")


def test_jumprelu_gate_gradcheck() -> None:
    """The torch autograd backward is consistent with its own forward.

    Runs without the compiled extension: it validates the straight-through logit
    gradient and the accumulated threshold gradient against finite differences of
    the pure-torch forward.
    """
    torch.manual_seed(0)
    rows, cols = 4, 6
    temperature = 0.37
    # Keep every logit clear of its threshold so the finite-difference probe
    # never straddles the hard ``1[l > θ]`` jump (the STE surrogate is smooth;
    # the returned value is not). Thresholds are positive; logits sit a full
    # unit above them.
    thresholds = torch.rand(cols, dtype=torch.float64, requires_grad=True)
    logits = (
        thresholds.detach() + 1.0 + torch.rand(rows, cols, dtype=torch.float64)
    ).requires_grad_(True)

    assert torch.autograd.gradcheck(
        lambda x, t: jumprelu_bounded_gate(x, t, temperature),
        (logits, thresholds),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
    )


def test_jumprelu_gate_matches_rust_row_by_row() -> None:
    """Torch forward value + both gradients match the Rust kernel to 1e-10."""
    rust = _rust_module()

    rng = np.random.default_rng(1234)
    rows, cols = 5, 7
    temperature = 0.41
    logits_np = rng.standard_normal((rows, cols))
    thresholds_np = rng.standard_normal(cols)

    logits = torch.tensor(logits_np, dtype=torch.float64, requires_grad=True)
    thresholds = torch.tensor(thresholds_np, dtype=torch.float64, requires_grad=True)

    value = jumprelu_bounded_gate(logits, thresholds, temperature)

    # Reference: run the Rust row kernel independently on each row.
    rust_value = np.empty_like(logits_np)
    rust_grad = np.empty_like(logits_np)
    thr = np.ascontiguousarray(thresholds_np)
    for r in range(rows):
        v_r, g_r = rust.sae_jumprelu_row_value_grad(
            np.ascontiguousarray(logits_np[r]), float(temperature), thr
        )
        rust_value[r] = np.asarray(v_r, dtype=np.float64)
        rust_grad[r] = np.asarray(g_r, dtype=np.float64)

    np.testing.assert_allclose(
        value.detach().cpu().numpy(),
        rust_value,
        rtol=0.0,
        atol=1e-10,
        err_msg="torch JumpReLU gate forward value must match the Rust kernel.",
    )

    # Logit gradient: seed the upstream with ones so each output cell contributes
    # its own diagonal ``da_k/dl_k`` — that is exactly the Rust ``grad`` row.
    upstream = torch.ones_like(value)
    (logit_grad, thr_grad) = torch.autograd.grad(value, (logits, thresholds), upstream)

    np.testing.assert_allclose(
        logit_grad.detach().cpu().numpy(),
        rust_grad,
        rtol=0.0,
        atol=1e-10,
        err_msg="torch JumpReLU logit gradient must match the Rust straight-through derivative.",
    )

    # Threshold gradient is the per-atom negation of the Rust derivative summed
    # over rows (``∂a_k/∂θ_k = −da_k/dl_k``); callers negate and accumulate.
    np.testing.assert_allclose(
        thr_grad.detach().cpu().numpy(),
        -rust_grad.sum(axis=0),
        rtol=0.0,
        atol=1e-10,
        err_msg="torch JumpReLU threshold gradient must be the negated row-sum of the Rust derivative.",
    )
