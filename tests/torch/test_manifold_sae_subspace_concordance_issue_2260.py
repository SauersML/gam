"""#2260: constrain angle-bearing harmonics and report cross-seed stability."""

from __future__ import annotations

import torch

from gamfit.torch import (
    ManifoldSAE,
    ManifoldSAEConfig,
    circular_concordance,
)


def _config(frame: torch.Tensor) -> ManifoldSAEConfig:
    return ManifoldSAEConfig(
        input_dim=6,
        n_atoms=1,
        intrinsic_rank=1,
        atom_manifold="circle",
        atom_basis="fourier",
        n_basis_per_atom=5,
        encoder_hidden=0,
        sparsity={"kind": "softmax_topk", "target_k": 1},
        decoder_subspace=frame,
        dtype=torch.float64,
    )


def _assert_harmonics_in_frame(model: ManifoldSAE) -> None:
    frame = model.decoder_subspace_frame
    assert frame is not None
    identity = torch.eye(frame.shape[1], dtype=frame.dtype, device=frame.device)
    orthogonal_complement = identity - frame.T @ frame
    out_of_plane = model.decoder_blocks[:, 1:, :] @ orthogonal_complement
    resolution = torch.finfo(frame.dtype).eps**0.5
    scale = torch.linalg.vector_norm(model.decoder_blocks[:, 1:, :]).clamp_min(1.0)
    assert torch.linalg.vector_norm(out_of_plane) <= resolution * scale


def test_circle_harmonics_have_only_live_subspace_coordinates() -> None:
    raw_frame = torch.tensor(
        [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 3.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    model = ManifoldSAE(_config(raw_frame))
    frame = model.decoder_subspace_frame
    assert frame is not None
    torch.testing.assert_close(
        frame @ frame.T,
        torch.eye(2, dtype=frame.dtype),
        rtol=torch.finfo(frame.dtype).eps**0.5,
        atol=torch.finfo(frame.dtype).eps**0.5,
    )

    # The optimizer owns exactly D ambient DC coefficients plus (K-1)*r live
    # harmonic coordinates. There is no full ambient harmonic parameter whose
    # orthogonal component is merely projected away.
    encoded = model.parametrizations.decoder_blocks.original
    assert tuple(encoded.shape) == (1, 6 + 4 * 2)
    _assert_harmonics_in_frame(model)

    generator = torch.Generator().manual_seed(2260)
    x = torch.randn(64, 6, generator=generator, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out = model(x)
    loss = (out.x_hat - x).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _assert_harmonics_in_frame(model)

    clone = ManifoldSAE(_config(raw_frame))
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone.decoder_blocks, model.decoder_blocks)


def test_circular_concordance_quotients_o2_and_rejects_collapsed_replica() -> None:
    n = 41
    row = torch.arange(n, dtype=torch.float64)
    base = torch.remainder((row.square() + 3.0 * row + 1.0) / n, 1.0)
    rotated = torch.remainder(base + 0.23, 1.0)
    reflected = torch.remainder(0.41 - base, 1.0)
    report = circular_concordance(torch.stack((base, rotated, reflected)))

    resolution = torch.finfo(torch.float64).eps**0.5
    assert all(item.well_posed for item in report.coverage)
    assert report.minimum_aligned_score is not None
    assert report.minimum_aligned_score >= 1.0 - resolution
    assert report.pairs[0].reflected is False
    assert report.pairs[1].reflected is True

    collapsed = circular_concordance(
        torch.stack((base, rotated, torch.zeros_like(base)))
    )
    assert collapsed.coverage[2].well_posed is False
    assert collapsed.pairs[1].aligned_score is None
    assert collapsed.minimum_aligned_score is None
    assert collapsed.mean_aligned_score is None
