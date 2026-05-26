"""RED tests for issue #227: GatedSAEDecoder must obey canonical Gated-SAE
Heaviside gating (gate active iff logit > 0).

Currently Python uses sigmoid(logit) > 0.5 (== logit > 0) while Rust uses
`logit != 0.0` (effectively always active). These tests pin the canonical
contract and will fail against the current Rust implementation.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")


def _heaviside_reference(x: np.ndarray, w_gate: np.ndarray, w_amp: np.ndarray) -> np.ndarray:
    """Canonical Gated-SAE decoder: gate := Heaviside(W_gate x), atoms := W_amp (gate * x)."""
    logits = x @ w_gate.T
    gates = (logits > 0.0).astype(np.float64)
    return (gates * x) @ w_amp.T


def test_python_decode_matches_heaviside_reference() -> None:
    from gamfit import GatedSAEDecoder

    rng = np.random.default_rng(11)
    w_gate = rng.standard_normal((5, 5))
    w_amp = rng.standard_normal((7, 5))
    x = rng.standard_normal((9, 5))

    expected = _heaviside_reference(x, w_gate, w_amp)
    got = GatedSAEDecoder(w_gate=w_gate, w_amp=w_amp).decode(x)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


def test_rust_decode_matches_heaviside_reference_negative_logits() -> None:
    """Rust currently gates on `logit != 0.0` so it admits negative logits.
    Canonical contract: negative logits must zero out their coordinate."""
    from gamfit._binding import rust_module

    # Construct W_gate so every logit is strictly negative for the given x.
    x = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    w_gate = -np.ones((3, 3), dtype=np.float64)  # logits = -3 each row
    w_amp = np.eye(3, dtype=np.float64)

    expected = _heaviside_reference(x, w_gate, w_amp)  # all zeros
    rust = np.asarray(rust_module().gated_sae_decode(x, w_gate, w_amp))
    np.testing.assert_allclose(rust, expected, rtol=0.0, atol=1e-12)
    assert np.allclose(rust, 0.0), "negative gate logits must produce zero output"


def test_rust_decode_zero_logit_is_inactive() -> None:
    """Exact-zero logit: Heaviside(0) := 0 (gate inactive). Rust currently
    treats `!= 0.0` as the gate, so a 0 logit is inactive there *only* by
    coincidence; pin the contract explicitly."""
    from gamfit._binding import rust_module

    x = np.array([[2.0, -3.0]], dtype=np.float64)
    w_gate = np.zeros((2, 2), dtype=np.float64)  # all logits exactly zero
    w_amp = np.eye(2, dtype=np.float64)

    rust = np.asarray(rust_module().gated_sae_decode(x, w_gate, w_amp))
    np.testing.assert_allclose(rust, np.zeros_like(x), rtol=0.0, atol=0.0)


def test_python_rust_parity_random_inputs() -> None:
    """Cross-implementation parity at random inputs that exercise both signs."""
    from gamfit import GatedSAEDecoder
    from gamfit._binding import rust_module

    rng = np.random.default_rng(7)
    w_gate = rng.standard_normal((6, 6))
    w_amp = rng.standard_normal((8, 6))
    x = rng.standard_normal((32, 6))

    py = GatedSAEDecoder(w_gate=w_gate, w_amp=w_amp).decode(x)
    rust = np.asarray(rust_module().gated_sae_decode(x, w_gate, w_amp))
    np.testing.assert_allclose(py, rust, rtol=0.0, atol=1e-12)


def test_python_rust_parity_float32_inputs() -> None:
    """Parity must also hold when inputs arrive as float32 (FFI must upcast)."""
    from gamfit import GatedSAEDecoder
    from gamfit._binding import rust_module

    rng = np.random.default_rng(19)
    w_gate = rng.standard_normal((4, 4)).astype(np.float32)
    w_amp = rng.standard_normal((5, 4)).astype(np.float32)
    x = rng.standard_normal((12, 4)).astype(np.float32)

    py = GatedSAEDecoder(w_gate=w_gate, w_amp=w_amp).decode(x)
    rust = np.asarray(rust_module().gated_sae_decode(x, w_gate, w_amp))
    np.testing.assert_allclose(py, rust, rtol=1e-6, atol=1e-6)
