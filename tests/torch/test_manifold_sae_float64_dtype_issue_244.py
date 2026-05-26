"""RED tests for issue #244 — `ManifoldSAE` must accept float64 input natively.

These tests fail today because `nn.Linear`, `atom_raw_anchor`, and
`decoder_blocks` are created with PyTorch's default dtype (float32), so
`forward(x.double())` hits a dtype mismatch in the encoder matmul.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import gamfit.torch as gt


def _cfg() -> gt.ManifoldSAEConfig:
    return gt.ManifoldSAEConfig(
        input_dim=5,
        n_atoms=2,
        intrinsic_rank=2,
        atom_manifold="cylinder",
        atom_basis="bspline",
        n_basis_per_atom=6,
    )


def test_manifold_sae_accepts_float64_input_without_manual_double() -> None:
    sae = gt.ManifoldSAE(_cfg())
    x = torch.randn(3, 5, dtype=torch.float64)
    out = sae(x)
    assert out.x_hat.dtype == torch.float64


def test_manifold_sae_float64_backward_works() -> None:
    sae = gt.ManifoldSAE(_cfg())
    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    out = sae(x)
    loss = out.x_hat.pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.dtype == torch.float64


def test_manifold_sae_default_dtype_is_float64() -> None:
    """The whole gamfit numerical stack is float64; the torch module should match."""
    sae = gt.ManifoldSAE(_cfg())
    assert sae.decoder_blocks.dtype == torch.float64
    assert sae.atom_raw_anchor.dtype == torch.float64
    enc = sae.encoder
    if isinstance(enc, torch.nn.Linear):
        assert enc.weight.dtype == torch.float64
    else:
        for m in enc:
            if isinstance(m, torch.nn.Linear):
                assert m.weight.dtype == torch.float64


def test_manifold_sae_config_accepts_explicit_dtype() -> None:
    """A `dtype` field should let callers pick float32 explicitly if they want it."""
    cfg = gt.ManifoldSAEConfig(
        input_dim=5,
        n_atoms=2,
        intrinsic_rank=2,
        atom_manifold="cylinder",
        atom_basis="bspline",
        n_basis_per_atom=6,
        dtype=torch.float32,
    )
    sae = gt.ManifoldSAE(cfg)
    assert sae.decoder_blocks.dtype == torch.float32
    x = torch.randn(3, 5, dtype=torch.float32)
    out = sae(x)
    assert out.x_hat.dtype == torch.float32
