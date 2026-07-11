"""Regression tests for the skip-transcoder sparsity objective.

``skip_transcoder(..., lambda_sparse=...)`` must actually reach the
module's objective — ``SkipAffineSmooth`` must carry ``lambda_sparse`` into its
``SmoothThresholdPenalty`` weight, and ``loss()`` must include the sparse penalty so
different ``lambda_sparse`` give a different objective on the same input.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from gamfit.torch.skip_transcoder import (
    SkipAffineSmooth,
    SkipTranscoderProfile,
    select_skip_transcoder,
    skip_transcoder,
)


def test_skip_affine_lambda_sparse_sets_penalty_weight():
    lo = SkipAffineSmooth(in_dim=4, out_dim=4, n_atoms=8, rank_skip=2, lambda_sparse=1e-3)
    hi = SkipAffineSmooth(in_dim=4, out_dim=4, n_atoms=8, rank_skip=2, lambda_sparse=1.0)
    assert lo.smooth_threshold.weight == pytest.approx(1e-3)
    assert hi.smooth_threshold.weight == pytest.approx(1.0)
    assert lo.smooth_threshold.weight != hi.smooth_threshold.weight


def test_skip_affine_different_lambda_gives_different_loss():
    torch.manual_seed(3)
    x = torch.randn(7, 4, dtype=torch.float64)
    y = torch.randn(7, 4, dtype=torch.float64)
    lo = SkipAffineSmooth(
        in_dim=4, out_dim=4, n_atoms=8, rank_skip=2, lambda_sparse=1e-3, dtype=torch.float64
    )
    hi = SkipAffineSmooth(
        in_dim=4, out_dim=4, n_atoms=8, rank_skip=2, lambda_sparse=1.0, dtype=torch.float64
    )
    # Identical weights so any loss difference is purely the sparse term.
    with torch.no_grad():
        hi.W_enc.copy_(lo.W_enc)
        hi.b_enc.copy_(lo.b_enc)
        hi.W_dec.copy_(lo.W_dec)
        hi.b_out.copy_(lo.b_out)
        hi.skip_U.copy_(lo.skip_U)
        hi.skip_V.copy_(lo.skip_V)
    loss_lo = lo.loss(x, y)
    loss_hi = hi.loss(x, y)
    assert torch.isfinite(loss_lo)
    assert torch.isfinite(loss_hi)
    assert not torch.isclose(loss_lo, loss_hi)
    # Larger lambda_sparse penalizes activations more -> larger total objective
    # (reconstruction term is identical because the weights match).
    assert float(loss_hi.item()) > float(loss_lo.item())


def test_skip_affine_loss_is_recon_plus_sparsity():
    torch.manual_seed(4)
    x = torch.randn(6, 3, dtype=torch.float64)
    y = torch.randn(6, 5, dtype=torch.float64)
    s = SkipAffineSmooth(
        in_dim=3, out_dim=5, n_atoms=7, rank_skip=2, lambda_sparse=0.5, dtype=torch.float64
    )
    y_hat, _ = s(x)
    recon = torch.mean((y_hat - y) ** 2)
    pen = s.sparsity_penalty(x)
    total = s.loss(x, y)
    assert torch.allclose(total, recon + pen)


def test_skip_transcoder_scalar_path_carries_lambda_sparse():
    s = skip_transcoder(in_dim=4, out_dim=4, n_atoms=8, rank_skip=2, lambda_sparse=2e-3)
    assert isinstance(s, SkipAffineSmooth)
    assert s.smooth_threshold.weight == pytest.approx(2e-3)


def test_skip_transcoder_rejects_cartesian_hyperparameter_sequences():
    with pytest.raises((TypeError, ValueError)):
        skip_transcoder(
            in_dim=4,
            out_dim=4,
            n_atoms=8,
            rank_skip=2,
            lambda_sparse=[1e-3, 1e-1],
        )


def test_skip_transcoder_continuous_profile_and_rank_birth_death_certificate():
    target_logs = (0.37, -0.61)

    def profile(trial):
        rank_cost = float((trial.rank_skip - 2) ** 2)
        dl = trial.log_lambda_sparse - target_logs[0]
        dt = trial.log_activation_threshold - target_logs[1]
        score = dl * dl + dt * dt + rank_cost
        smooth = SkipAffineSmooth(
            in_dim=3,
            out_dim=3,
            n_atoms=4,
            rank_skip=trial.rank_skip,
            lambda_sparse=math.exp(trial.log_lambda_sparse),
            activation_threshold=math.exp(trial.log_activation_threshold),
            dtype=torch.float64,
        )
        return SkipTranscoderProfile(
            smooth=smooth,
            rank_skip=trial.rank_skip,
            log_lambda_sparse=trial.log_lambda_sparse,
            log_activation_threshold=trial.log_activation_threshold,
            negative_log_evidence=score,
            gradient_log_hyperparameters=(2.0 * dl, 2.0 * dt),
            gradient_scale=1.0 + abs(score),
        )

    selected = select_skip_transcoder(
        3,
        3,
        profile,
        initial_lambda_sparse=0.2,
        initial_activation_threshold=0.8,
    )
    assert selected.rank_skip == 2
    assert selected.profile.log_lambda_sparse == pytest.approx(target_logs[0], abs=1e-7)
    assert selected.profile.log_activation_threshold == pytest.approx(
        target_logs[1], abs=1e-7
    )
    assert selected.certificate.death.structurally_feasible
    assert selected.certificate.birth.structurally_feasible
    assert selected.certificate.death.gap_from_selected > 0.0
    assert selected.certificate.birth.gap_from_selected > 0.0
    assert all(
        transition.to_rank in {0, 1, 2, 3}
        for transition in selected.certificate.transitions
    )
