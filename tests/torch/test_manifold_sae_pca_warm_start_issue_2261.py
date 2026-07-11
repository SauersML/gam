"""#2261: the curved torch arm starts at its exact converged PCA submodel."""

from __future__ import annotations

import torch

from gamfit._binding import rust_module
from gamfit.torch import ManifoldSAE, ManifoldSAEConfig


def _data() -> torch.Tensor:
    generator = torch.Generator().manual_seed(2261)
    scores = torch.randn(96, 6, generator=generator, dtype=torch.float64)
    mixing = torch.randn(6, 8, generator=generator, dtype=torch.float64)
    scales = torch.tensor(
        [4.0, 2.5, 1.7, 0.9, 0.4, 0.15], dtype=torch.float64
    )
    mean = torch.tensor(
        [1.5, -0.8, 0.3, 2.0, -1.2, 0.7, 0.1, -0.4], dtype=torch.float64
    )
    return (scores * scales) @ mixing + mean


def _model() -> ManifoldSAE:
    return ManifoldSAE(
        ManifoldSAEConfig(
            input_dim=8,
            n_atoms=3,
            intrinsic_rank=1,
            atom_manifold="circle",
            atom_basis="fourier",
            n_basis_per_atom=5,
            encoder_hidden=0,
            sparsity={"kind": "softmax_topk", "target_k": 3},
            dtype=torch.float64,
        )
    )


def _pca_projection(x: torch.Tensor, rank: int) -> torch.Tensor:
    mean_np, basis_np = rust_module().sae_principal_subspace(
        x.detach().cpu().numpy(), rank
    )
    mean = torch.as_tensor(mean_np, dtype=x.dtype)
    basis = torch.as_tensor(basis_np, dtype=x.dtype)
    centered = x - mean
    return mean + (centered @ basis.T) @ basis


def test_first_training_forward_is_exact_pca_and_curvature_remains_trainable() -> None:
    x = _data()
    model = _model()
    model.train()

    out = model(x)
    expected = _pca_projection(x, rank=3)

    assert bool(model._pca_initialized.item())
    torch.testing.assert_close(out.x_hat, expected, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(
        model.decoder_blocks[:, 1:, :],
        torch.zeros_like(model.decoder_blocks[:, 1:, :]),
        rtol=0.0,
        atol=0.0,
    )

    loss = (out.x_hat - x).square().mean()
    loss.backward()
    harmonic_grad = model.decoder_blocks.grad
    assert harmonic_grad is not None
    assert torch.linalg.vector_norm(harmonic_grad[:, 1:, :]).item() > 0.0


def test_full_corpus_seed_persists_mean_and_frame_across_state_dict() -> None:
    x = _data()
    model = _model()
    model.initialize_from_pca(x)
    model.eval()
    expected = _pca_projection(x, rank=3)

    clone = _model()
    clone.load_state_dict(model.state_dict())
    clone.eval()

    assert bool(clone._pca_initialized.item())
    torch.testing.assert_close(clone._linear_mean, model._linear_mean)
    torch.testing.assert_close(clone(x).x_hat, expected, rtol=2e-12, atol=2e-12)


def test_sparse_dictionary_cannot_be_mislabeled_as_exact_pca() -> None:
    x = _data()
    model = ManifoldSAE(
        ManifoldSAEConfig(
            input_dim=8,
            n_atoms=3,
            intrinsic_rank=1,
            atom_manifold="circle",
            atom_basis="fourier",
            n_basis_per_atom=5,
            encoder_hidden=0,
            sparsity={"kind": "softmax_topk", "target_k": 2},
            dtype=torch.float64,
        )
    )

    try:
        model.initialize_from_pca(x)
    except ValueError as error:
        assert "target_k == n_atoms" in str(error)
    else:
        raise AssertionError("a sparse top-k dictionary is not an exact PCA projection")
