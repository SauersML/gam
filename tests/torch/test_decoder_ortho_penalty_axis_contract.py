"""RED tests for issue #245: ManifoldSAE.decoder_ortho_penalty() axis contract.

The torch wrapper builds row-block groups over flattened decoder rows but the
Rust BlockOrthogonalityPenalty validates `axis < latent_dim` against the
*column* axis. So any `cfg.decoder.ortho_weight > 0` config with F >= 2 and
K >= 1 raises ValueError today. These tests pin the public contract: a
configured ortho penalty must (a) be callable without error, (b) return a
finite scalar tensor, (c) be non-constant in the decoder, and (d) be
differentiable w.r.t. `decoder_blocks`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")


def _make_sae(F: int, K: int, D: int, weight: float = 1e-2):
    cfg = gt.ManifoldSAEConfig(
        input_dim=D,
        n_atoms=F,
        intrinsic_rank=1,
        n_basis_per_atom=K,
        decoder={"ortho_weight": weight},
    )
    return gt.ManifoldSAE(cfg).double()


def test_decoder_ortho_penalty_does_not_raise() -> None:
    """Issue-245 repro: weight>0 with F>=2 must not raise on call."""
    sae = _make_sae(F=3, K=4, D=4)
    out = sae.decoder_ortho_penalty()
    assert torch.is_tensor(out)
    assert out.ndim == 0
    assert torch.isfinite(out)


def test_decoder_ortho_penalty_is_scalar_and_finite_across_shapes() -> None:
    """Sweep shapes that all currently raise."""
    for F, K, D in [(2, 1, 1), (2, 3, 5), (4, 2, 7), (3, 5, 2)]:
        sae = _make_sae(F=F, K=K, D=D)
        out = sae.decoder_ortho_penalty()
        assert out.ndim == 0, f"non-scalar output for F={F},K={K},D={D}"
        assert torch.isfinite(out), f"non-finite for F={F},K={K},D={D}"


def test_decoder_ortho_penalty_varies_with_decoder_blocks() -> None:
    """The penalty must depend on decoder state — otherwise it is a no-op."""
    sae = _make_sae(F=3, K=4, D=4)
    with torch.no_grad():
        sae.decoder_blocks.zero_()
        sae.decoder_blocks[0, 0, 0] = 1.0
        sae.decoder_blocks[1, 0, 0] = 1.0
    val_aligned = sae.decoder_ortho_penalty().item()

    with torch.no_grad():
        sae.decoder_blocks.zero_()
        sae.decoder_blocks[0, 0, 0] = 1.0
        sae.decoder_blocks[1, 0, 1] = 1.0
    val_orthogonal = sae.decoder_ortho_penalty().item()

    assert val_aligned != val_orthogonal, (
        "ortho penalty must distinguish aligned vs orthogonal atoms; "
        f"got aligned={val_aligned} orthogonal={val_orthogonal}"
    )
    assert val_aligned > val_orthogonal, (
        "aligned atoms should penalize MORE than orthogonal ones; "
        f"got aligned={val_aligned} orthogonal={val_orthogonal}"
    )


def test_decoder_ortho_penalty_is_differentiable() -> None:
    """Gradient w.r.t. decoder_blocks must exist and be non-trivially nonzero."""
    sae = _make_sae(F=3, K=4, D=4)
    sae.decoder_blocks.requires_grad_(True)
    with torch.no_grad():
        torch.manual_seed(0)
        sae.decoder_blocks.normal_(0.0, 0.3)
    out = sae.decoder_ortho_penalty()
    out.backward()
    g = sae.decoder_blocks.grad
    assert g is not None
    assert torch.isfinite(g).all()
    assert g.abs().sum().item() > 0.0


def test_decoder_ortho_penalty_zero_when_perfectly_orthogonal() -> None:
    """When all per-atom flattened weight vectors are mutually orthogonal,
    the cross-block Frobenius sum should be (numerically) zero."""
    F, K, D = 3, 1, 4
    sae = _make_sae(F=F, K=K, D=D)
    with torch.no_grad():
        sae.decoder_blocks.zero_()
        # one nonzero coordinate per atom, all distinct -> mutually orthogonal
        sae.decoder_blocks[0, 0, 0] = 1.0
        sae.decoder_blocks[1, 0, 1] = 1.0
        sae.decoder_blocks[2, 0, 2] = 1.0
    val = sae.decoder_ortho_penalty().item()
    assert val == pytest.approx(0.0, abs=1e-10), (
        f"perfectly orthogonal atoms should give 0 penalty; got {val}"
    )
