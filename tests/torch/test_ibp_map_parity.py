"""Value/gradient contract for the Rust-backed IBP-MAP relaxation.

The torch IBP-MAP map calls the Rust source of truth
``gam::terms::sae::assignment::ordered_beta_bernoulli_batch_value_grad``. This test pins the
Torch bridge's forward value and diagonal logit Jacobian to the Rust kernel and
exercises the cached-Jacobian autograd backward.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")

from gamfit.torch.penalties import ordered_beta_bernoulli  # noqa: E402


def _rust_module() -> Any:
    """Return the compiled ``gamfit._rust`` extension, or skip if unavailable."""
    binding = importlib.import_module("gamfit._binding")
    try:
        return binding.rust_module()
    except Exception as exc:  # RustExtensionUnavailableError on pure-Python shims
        pytest.skip(f"compiled gamfit._rust extension unavailable: {exc}")


def test_ordered_beta_bernoulli_gradcheck() -> None:
    """The torch autograd backward is consistent with its own forward.

    Validates the diagonal logit Jacobian against finite differences of the
    Rust-backed forward.
    """
    torch.manual_seed(0)
    rows, cols = 4, 6
    logits = torch.randn(rows, cols, dtype=torch.float64, requires_grad=True)
    temperature = 0.37
    assert torch.autograd.gradcheck(
        lambda x: ordered_beta_bernoulli(x, temperature),
        (logits,),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
    )


def _assert_matches_rust(rows: int, cols: int, temperature: float) -> None:
    rust = _rust_module()

    rng = np.random.default_rng(1234)
    logits_np = rng.standard_normal((rows, cols))
    logits = torch.tensor(logits_np, dtype=torch.float64, requires_grad=True)

    value = ordered_beta_bernoulli(logits, temperature)

    # Reference: run the Rust row kernel independently on each row.
    rust_value = np.empty_like(logits_np)
    rust_grad = np.empty_like(logits_np)
    for r in range(rows):
        v_r, g_r = rust.sae_ordered_beta_bernoulli_value_grad(
            np.ascontiguousarray(logits_np[r]), float(temperature)
        )
        rust_value[r] = np.asarray(v_r, dtype=np.float64)
        rust_grad[r] = np.asarray(g_r, dtype=np.float64)

    np.testing.assert_allclose(
        value.detach().cpu().numpy(),
        rust_value,
        rtol=0.0,
        atol=1e-9,
        err_msg="torch IBP-MAP forward value must match the Rust kernel.",
    )

    # Logit gradient: seed the upstream with ones so each output cell contributes
    # its own diagonal ``dz_k/dl_k`` — that is exactly the Rust ``grad`` row.
    upstream = torch.ones_like(value)
    (logit_grad,) = torch.autograd.grad(value, (logits,), upstream)

    np.testing.assert_allclose(
        logit_grad.detach().cpu().numpy(),
        rust_grad,
        rtol=0.0,
        atol=1e-9,
        err_msg="torch IBP-MAP logit Jacobian must match the Rust diagonal derivative.",
    )


def test_ordered_beta_bernoulli_matches_rust_row_by_row() -> None:
    """Torch forward value + diagonal Jacobian match the Rust kernel to 1e-9."""
    _assert_matches_rust(rows=5, cols=7, temperature=0.41)


def test_ordered_beta_bernoulli_matches_rust_large_k() -> None:
    """The allocation-free posterior-mean kernel remains exact at large K."""
    _assert_matches_rust(rows=3, cols=400, temperature=0.5)
