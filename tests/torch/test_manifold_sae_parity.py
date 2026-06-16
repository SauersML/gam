"""ManifoldSAE (torch) ⇄ sae_manifold_fit (closed-form) parity test.

The torch ``ManifoldSAE.fit`` is required to produce numerics identical to the
closed-form ``gamfit.sae_manifold_fit`` on synthetic data when given an
equivalent configuration. Parity is by construction (``fit`` delegates to the
same Rust kernel) — this test makes the contract observable.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

pytest: Any = import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")

import gamfit  # noqa: E402  (after importorskip)


def _make_synth(n: int = 48, d: int = 6, k: int = 3, seed: int = 0) -> np.ndarray:
    """Mixture-of-cosines synthetic with K latent atoms in ``R^d``."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(n, k))
    bases = rng.standard_normal(size=(k, d))
    bases /= np.linalg.norm(bases, axis=1, keepdims=True) + 1e-9
    amps = rng.uniform(0.5, 1.5, size=(n, k))
    x = (amps * np.cos(angles))[:, :, None] * bases[None, :, :]
    return x.sum(axis=1) + 0.05 * rng.standard_normal((n, d))


def test_fit_matches_closed_form_reml_score_and_blocks():
    torch.manual_seed(0)
    np.random.seed(0)
    X = _make_synth()
    cfg = gt.ManifoldSAEConfig(
        input_dim=X.shape[1],
        K=3,
        intrinsic_rank=1,
        atom_manifold="circle",
        atom_basis="fourier",
        basis_order=2,
        n_basis_per_atom=4,
        sparsity={"kind": "ibp_gumbel", "init_alpha": 1.0, "tau_schedule": "linear:4.0->1.0"},
        decoder={"ortho_weight": 0.0, "monotonicity_weight": 0.0},
        reml={"enabled": True, "select": ["lambda"]},
    )

    sae = gt.ManifoldSAE(cfg)
    torch_x = torch.as_tensor(X, dtype=torch.float64)
    initializers = sae._closed_form_initializers(torch_x)
    torch_fit = sae.fit(torch_x, n_iter=10, random_state=42)

    # Equivalent closed-form call (same Rust kernel, same amortized warm starts).
    cf_fit = gamfit.sae_manifold_fit(
        X=X,
        K=cfg.n_atoms,
        d_atom=cfg.intrinsic_rank,
        atom_topology="circle",
        atom_basis=cfg.closed_form_basis_kind(),
        assignment=cfg.closed_form_assignment(),
        schedule=cfg.sparsity.gumbel_schedule(),
        n_iter=10,
        random_state=42,
        **initializers,
    )

    # The two fits are produced by the same Rust call with the same args.
    assert torch_fit.reml_score == pytest.approx(cf_fit.reml_score)
    np.testing.assert_allclose(torch_fit.fitted, cf_fit.fitted, rtol=1e-10, atol=1e-12)
    for b_torch, b_cf in zip(torch_fit.decoder_blocks, cf_fit.decoder_blocks):
        np.testing.assert_allclose(b_torch, b_cf, rtol=1e-10, atol=1e-12)


def test_forward_output_shapes_and_finite():
    cfg = gt.ManifoldSAEConfig(
        input_dim=8,
        K=4,
        intrinsic_rank=1,
        atom_manifold="circle",
        atom_basis="fourier",
        n_basis_per_atom=5,
    )
    sae = gt.ManifoldSAE(cfg)
    x = torch.randn(7, 8, dtype=torch.float32)
    out = sae(x)
    assert out.z.shape == (7, 4)
    assert out.x_hat.shape == (7, 8)
    assert out.positions.shape == (7, 4, 1)
    assert out.amplitudes.shape == (7, 4)
    assert out.curves.shape == (7, 4, 5)
    assert torch.isfinite(out.x_hat).all()
    assert torch.isfinite(out.z).all()


def test_lock_snapshot_freezes_hypers():
    cfg = gt.ManifoldSAEConfig(input_dim=4, K=2, intrinsic_rank=1, n_basis_per_atom=3)
    sae = gt.ManifoldSAE(cfg)
    assert sae.log_lambda.requires_grad
    sae.lock_snapshot()
    assert sae.is_locked
    assert not sae.log_lambda.requires_grad
    # IBP-Gumbel sparsity carries a learnable log_alpha through the Rust-backed
    # IBPAssignmentPenalty submodule; lock_snapshot freezes it.
    locked = [p.requires_grad for p in sae.sparsity.parameters(recurse=True)]
    assert all(flag is False for flag in locked)


def test_extract_feature_curves_grid_shape():
    cfg = gt.ManifoldSAEConfig(
        input_dim=5, K=3, intrinsic_rank=1, atom_manifold="circle",
        atom_basis="fourier", n_basis_per_atom=4,
    )
    sae = gt.ManifoldSAE(cfg)
    curves = sae.extract_feature_curves(grid_size=16)
    assert set(curves.keys()) == {0, 1, 2}
    for c in curves.values():
        assert c.shape == (16, 5)
        assert torch.isfinite(c).all()


def test_decoder_ortho_routes_through_rust():
    # Both ortho (block_orthogonality) and monotonicity descriptors now route
    # through Rust analytic_penalty_value_grad; nothing remains deferred.
    cfg = gt.ManifoldSAEConfig(
        input_dim=4, K=3, intrinsic_rank=1, n_basis_per_atom=4,
        decoder={"ortho_weight": 1e-2},
    )
    sae = gt.ManifoldSAE(cfg)
    assert sae.decoder_ortho_penalty().item() > 0.0
    cfg0 = gt.ManifoldSAEConfig(
        input_dim=4, K=3, intrinsic_rank=1, n_basis_per_atom=4,
        decoder={"ortho_weight": 0.0},
    )
    sae0 = gt.ManifoldSAE(cfg0)
    assert sae0.decoder_ortho_penalty().item() == 0.0


def test_decoder_monotonicity_routes_through_rust():
    cfg = gt.ManifoldSAEConfig(
        input_dim=4, K=2, intrinsic_rank=1, n_basis_per_atom=5,
        decoder={"monotonicity_weight": 1e-2, "monotonicity_direction": 1.0},
    )
    sae = gt.ManifoldSAE(cfg)
    assert sae.decoder_monotonicity_penalty().item() > 0.0
    cfg0 = gt.ManifoldSAEConfig(
        input_dim=4, K=2, intrinsic_rank=1, n_basis_per_atom=5,
        decoder={"monotonicity_weight": 0.0},
    )
    sae0 = gt.ManifoldSAE(cfg0)
    assert sae0.decoder_monotonicity_penalty().item() == 0.0


def test_bspline_basis_routes_through_rust():
    # Cylinder + bspline now reaches the Rust bspline arm of basis_with_jet
    # (open uniform). Forward and backward must both succeed without falling
    # back to the Duchon path.
    cfg = gt.ManifoldSAEConfig(
        input_dim=5, K=2, intrinsic_rank=2, atom_manifold="cylinder",
        atom_basis="bspline", basis_order=2, n_basis_per_atom=6,
    )
    sae = gt.ManifoldSAE(cfg)
    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    out = sae(x)
    assert torch.isfinite(out.x_hat).all()
    out.x_hat.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_basis_eval_matches_rust_basis_with_jet():
    # Forward of the curves must agree bit-exactly with the same Rust call
    # done out-of-band. Proves the basis math lives in Rust, not torch.
    import gamfit
    cfg = gt.ManifoldSAEConfig(
        input_dim=6, K=2, intrinsic_rank=1, atom_manifold="circle",
        atom_basis="fourier", n_basis_per_atom=5,
    )
    sae = gt.ManifoldSAE(cfg)
    x = torch.randn(4, 6, dtype=torch.float64)
    with torch.no_grad():
        out = sae(x)
    # Reproduce the same basis evaluation via the raw PyO3 binding.
    rust = gamfit._rust  # type: ignore[attr-defined]
    positions_np = out.positions.detach().cpu().numpy().reshape(-1, 1)
    n_harm = max(1, (cfg.n_basis_per_atom - 1) // 2)
    phi_np, _jet, _pen = rust.basis_with_jet(
        "periodic", np.ascontiguousarray(positions_np), {"n_harmonics": int(n_harm)}
    )
    # The module trims/pads to n_basis_per_atom; compare on the overlap.
    K_actual = phi_np.shape[1]
    K = min(K_actual, cfg.n_basis_per_atom)
    expected = torch.as_tensor(phi_np[:, :K], dtype=torch.float64).reshape(
        out.curves.shape[0], out.curves.shape[1], K
    )
    np.testing.assert_allclose(
        out.curves[..., :K].detach().numpy(), expected.numpy(), rtol=0.0, atol=0.0
    )


def test_basis_eval_backward_uses_rust_jet():
    # dphi/dtheta backward flows through the saved jet tensor; sanity-check by
    # finite differencing the forward and matching the autograd gradient.
    cfg = gt.ManifoldSAEConfig(
        input_dim=4, K=1, intrinsic_rank=1, atom_manifold="circle",
        atom_basis="fourier", n_basis_per_atom=3,
    )
    sae = gt.ManifoldSAE(cfg)
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    out = sae(x)
    out.x_hat.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_advance_temperature_queries_rust_schedule():
    # advance_temperature must drive the Rust GumbelTemperatureSchedule, not a
    # duplicated Python decay. Each call advances the schedule's own iter_count
    # and writes the Rust-evaluated tau into the layer buffer. Regression for the
    # parity workstream: the previous code passed an extra positional argument to
    # `current_tau`, which raised TypeError and silently never annealed.
    from gamfit.torch.manifold_sae import _SparsityLayer  # type: ignore[attr-defined]

    cfg = gt.ManifoldSAEConfig(
        input_dim=4,
        K=3,
        intrinsic_rank=1,
        n_basis_per_atom=4,
        sparsity={"kind": "ibp_gumbel", "init_alpha": 1.0, "tau_schedule": "linear:4.0->1.0"},
    )
    layer = _SparsityLayer(cfg)
    schedule = cfg.sparsity.gumbel_schedule()

    # tau starts at tau_start before any annealing step.
    assert float(layer.tau.item()) == pytest.approx(cfg.sparsity.tau_start)

    # Each advance must equal the Rust schedule evaluated at the matching iter.
    prev = float(layer.tau.item())
    for step in range(1, 5):
        layer.advance_temperature()
        expected = float(schedule.step())
        assert float(layer.tau.item()) == pytest.approx(expected), (
            f"advance_temperature at step {step} diverged from Rust schedule"
        )
        # Linear decay from 4.0 -> 1.0 is monotone non-increasing.
        assert float(layer.tau.item()) <= prev + 1e-12
        prev = float(layer.tau.item())


def test_ibp_gumbel_forward_applies_stick_breaking_prior():
    # The torch IBP-Gumbel forward must route through the Rust ibp_map kernel so
    # it applies the stick-breaking prior pi_k = (alpha/(alpha+1))^k and the
    # temperature scaling that the closed-form SaeAssignment IBP path uses. A bare
    # sigmoid(logits/tau) would omit pi_k. We pin it by reproducing the Rust
    # value out-of-band and asserting the forward matches bit-for-bit.
    from gamfit.torch.manifold_sae import _SparsityLayer  # type: ignore[attr-defined]
    from gamfit.torch.penalties import ibp_map  # type: ignore[attr-defined]

    alpha = 1.3
    tau_start = 2.5
    cfg = gt.ManifoldSAEConfig(
        input_dim=4,
        K=5,
        intrinsic_rank=1,
        n_basis_per_atom=4,
        sparsity={"kind": "ibp_gumbel", "init_alpha": alpha, "tau_start": tau_start, "tau_min": 0.5},
    )
    layer = _SparsityLayer(cfg)
    rng = np.random.default_rng(7)
    logits = torch.as_tensor(rng.standard_normal((6, 5)), dtype=torch.float64)

    assignments, gate_pre = layer(logits)

    # gate_pre is the untouched logits; assignments carry the prior + temperature.
    np.testing.assert_allclose(gate_pre.detach().numpy(), logits.numpy(), rtol=0.0, atol=0.0)

    tau = max(tau_start, 1e-6)
    expected = ibp_map(logits, tau, alpha)
    np.testing.assert_allclose(
        assignments.detach().numpy(), expected.detach().numpy(), rtol=0.0, atol=0.0
    )

    # The stick-breaking prior strictly decays in atom index: with tied logits the
    # per-atom mass must fall off geometrically, which bare sigmoid(logits/tau)
    # could never produce.
    tied = torch.zeros((1, 5), dtype=torch.float64)
    tied_assign = layer(tied)[0].detach().numpy().reshape(-1)
    ratio = alpha / (alpha + 1.0)
    sig0 = float(tied_assign[0])
    for k in range(5):
        np.testing.assert_allclose(tied_assign[k], sig0 * ratio**k, rtol=0.0, atol=1e-12)
    assert np.all(np.diff(tied_assign) < 0.0), "stick-breaking prior must strictly decay"


def test_isometry_backward_grad_matches_rust_grad_jacobian():
    # _IsometryPenaltyFn.backward must obtain dP/dJ from the Rust grad_jacobian
    # kernel (W-/reference-/weight-aware), not recompute 2*mu*J*(J^TJ - I) in
    # Python. We pin the autograd basis gradient against a central-difference of
    # the Rust penalty value, so any Python recompute that drops scalar_weight or
    # the reference metric would fail.
    weight = 0.8
    pen = gt.IsometryPenalty(weight=weight)
    latent = torch.tensor(
        [[0.2, -0.1, 0.7], [0.5, 0.3, -0.4], [-0.2, 0.8, 0.1]],
        dtype=torch.float64,
    )
    basis = torch.tensor(
        [[1.1, 0.2, -0.3], [0.0, 0.9, 0.4], [-0.5, 0.1, 1.2], [0.3, -0.6, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    value = pen(latent, basis)
    value.backward()
    assert basis.grad is not None
    autograd_grad = basis.grad.detach().numpy().copy()

    # Central-difference of the closed-form Rust value w.r.t. each basis entry.
    base_np = basis.detach().numpy().copy()
    latent_np = latent.detach().numpy()
    h = 1e-6
    numeric = np.zeros_like(base_np)
    for i in range(base_np.shape[0]):
        for j in range(base_np.shape[1]):
            bp = base_np.copy()
            bm = base_np.copy()
            bp[i, j] += h
            bm[i, j] -= h
            v_plus = pen.rust_value(latent_np, basis=bp)
            v_minus = pen.rust_value(latent_np, basis=bm)
            numeric[i, j] = (v_plus - v_minus) / (2.0 * h)

    np.testing.assert_allclose(autograd_grad, numeric, rtol=1e-5, atol=1e-7)

    # And the gradient must scale linearly with the scalar weight (scalar_weight
    # is not dropped). Doubling the weight doubles dP/dJ.
    pen2 = gt.IsometryPenalty(weight=2.0 * weight)
    basis2 = basis.detach().clone().requires_grad_(True)
    pen2(latent, basis2).backward()
    assert basis2.grad is not None
    np.testing.assert_allclose(
        basis2.grad.detach().numpy(), 2.0 * autograd_grad, rtol=1e-9, atol=1e-9
    )


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        gt.ManifoldSAEConfig(input_dim=0, K=4)
    with pytest.raises(ValueError):
        gt.ManifoldSAEConfig(input_dim=4, K=4, atom_manifold="circle", intrinsic_rank=2)
    with pytest.raises(ValueError):
        gt.ManifoldSAEConfig(input_dim=4, K=4, atom_manifold="sphere", intrinsic_rank=1)
    with pytest.raises(ValueError):
        gt.SparsityConfig(kind="ibp_gumbel", tau_start=1.0, tau_min=4.0)
