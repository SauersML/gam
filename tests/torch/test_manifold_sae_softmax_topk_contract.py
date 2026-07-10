"""softmax_topk ``target_k`` contract tests.

Relocated verbatim from ``gamfit/torch/manifold_sae.py`` (they lived there as
``__main__`` self-tests): the production module must stay a thin wrapper with
no test logic, and pytest runs these instead of a hand-invoked ``__main__``.
"""

from __future__ import annotations

import inspect

import torch

from gamfit.torch import manifold_sae as torch_manifold_sae
from gamfit.torch import penalties as torch_penalties
from gamfit.torch.manifold_sae import ManifoldSAE, ManifoldSAEConfig, SparsityConfig


def _selftest_softmax_topk_multi_atom_honors_target_k() -> None:
    """The multi-atom (``target_k > 1``) gate honors ``target_k`` up to ``n_atoms``.

    Builds a small ``ManifoldSAE`` (``n_atoms=64``) and, for a sweep of
    ``target_k`` values spanning the sparse regime (4, 8, 16, 32) and the
    formerly-capped dense regime (48, 64), measures the mean number of active
    atoms per row (atoms with a nonzero gated code ``z``) after a forward pass.

    Pre-fix behavior: target_k 4..32 were honored but 48 and 64 both plateaued at
    ~36/64 active because the non-negative ``code`` clamp zeroed every selected
    atom whose curve projected negatively onto the row. Post-fix: the multi-atom
    branch scores and gates with the *signed* least-squares code, so every
    selected atom carries magnitude and the active count tracks
    ``min(target_k, n_atoms)`` for every ``target_k`` — including 48 and 64.

    This branch is a pure hard top-k of the signed residual with no temperature
    dependence, so the count is honored in ``eval()`` at any ``tau`` (no training
    needed). Raises ``AssertionError`` on regression.
    """
    torch.manual_seed(0)
    n_atoms = 64
    input_dim = 32
    n_rows = 128

    measured: dict[int, float] = {}
    for target_k in (4, 8, 16, 32, 48, 64):
        cfg = ManifoldSAEConfig(
            input_dim=input_dim,
            n_atoms=n_atoms,
            intrinsic_rank=2,
            atom_manifold="product",
            atom_basis="duchon",
            sparsity=SparsityConfig(kind="softmax_topk", target_k=target_k),
            dtype=torch.float64,
        )
        model = ManifoldSAE(cfg)
        model.eval()
        x = torch.randn(n_rows, input_dim, dtype=torch.float64)
        with torch.no_grad():
            out = model(x)
        active_per_row = (out.z != 0).sum(dim=1).to(torch.float64)
        mean_active = float(active_per_row.mean().item())
        measured[target_k] = mean_active

        assert torch.isfinite(out.x_hat).all(), (
            f"reconstruction produced non-finite values at target_k={target_k}"
        )
        # The gate selects exactly min(target_k, n_atoms) atoms and every selected
        # atom now carries a (signed) nonzero code, so the active count should
        # equal target_k to numerical tolerance (a selected atom is zero only on
        # the measure-zero event recon·x == 0 exactly).
        expected = min(target_k, n_atoms)
        assert abs(mean_active - expected) <= 0.5, (
            f"target_k={target_k}: mean active/row={mean_active:.3f}, "
            f"expected ~{expected} (gate not honoring target_k)"
        )

    # Explicitly assert the formerly-capped dense regime no longer plateaus ~36.
    for target_k in (48, 64):
        assert measured[target_k] > 40.0, (
            f"dense regime regressed: target_k={target_k} gave "
            f"{measured[target_k]:.3f} active/row (old cap was ~36.2)"
        )

    summary = "  ".join(f"k={k}->{v:.3f}" for k, v in measured.items())
    print(f"softmax_topk (k>1) active/row vs target_k: {summary}")
    print("OK: softmax_topk gate honors target_k up to n_atoms (no ~36/64 cap).")


def _selftest_softmax_topk_k1_converges_to_one_active() -> None:
    """The ``target_k == 1`` router converges to ~1 active atom/row when annealed.

    The ``target_k == 1`` path is a deterministic-annealing (DA) EM router
    (issue #1282): its forward is a soft→hard interpolation controlled by
    ``progress = (tau_start - tau) / (tau_start - tau_min)``. At the schedule
    START (``tau = tau_start``, ``progress = 0``) the forward is deliberately the
    *soft* responsibility-weighted code — a near-uniform, dense gate — so the
    M-step can differentiate the atoms from a near-symmetric init instead of
    latching a random top-1 winner and collapsing (the exact #1282 failure). So
    an untrained model at ``tau_start`` reports many active atoms *by design*;
    that is the DA warmup, not a cap.

    The honest bar for a top-1 gate is the ANNEALED schedule that every real
    training run reaches: as ``tau -> tau_min`` (``progress -> 1``) the forward
    interpolates to the committed hard top-1 winner — exactly one active atom per
    row. This test trains a few Adam steps while advancing the temperature to
    ``tau_min`` and asserts the mean active count collapses to ~1, proving the
    ~1-active contract is honored without disturbing the DA warmup / anchor / EMA
    machinery that #1282 relies on.
    """
    torch.manual_seed(0)
    input_dim = 16
    n_atoms = 8
    n_rows = 96
    tau_steps = 40

    cfg = ManifoldSAEConfig(
        input_dim=input_dim,
        n_atoms=n_atoms,
        intrinsic_rank=2,
        atom_manifold="product",
        atom_basis="duchon",
        sparsity=SparsityConfig(
            kind="softmax_topk", target_k=1, tau_start=4.0, tau_min=1.0, tau_steps=tau_steps
        ),
        dtype=torch.float64,
    )
    model = ManifoldSAE(cfg)
    x = torch.randn(n_rows, input_dim, dtype=torch.float64)

    # Warmup count at the schedule start (soft DA phase) — documented, not asserted
    # to be ~1: this is the intended dense warmup.
    model.eval()
    with torch.no_grad():
        warm = float((model(x).z != 0).sum(dim=1).to(torch.float64).mean().item())

    # Train a few steps while annealing tau -> tau_min (a completed schedule).
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(120):
        opt.zero_grad()
        out = model(x)
        loss = ((out.x_hat - x) ** 2).mean()
        loss.backward()
        opt.step()
        model.sparsity.advance_temperature()

    model.eval()
    with torch.no_grad():
        out = model(x)
    mean_active = float((out.z != 0).sum(dim=1).to(torch.float64).mean().item())
    tau_now = float(model.sparsity.tau.item())

    assert torch.isfinite(out.x_hat).all(), "k=1 reconstruction produced non-finite values"
    assert abs(tau_now - cfg.sparsity.tau_min) < 1e-9, (
        f"schedule did not reach tau_min (tau={tau_now}); the honest k=1 bar is annealed"
    )
    assert abs(mean_active - 1.0) <= 0.25, (
        f"target_k=1 did not converge to ~1 active/row: mean active/row={mean_active:.3f} "
        f"(annealed, tau={tau_now})"
    )

    print(
        f"softmax_topk k=1 active/row: warmup(tau_start)={warm:.3f} -> "
        f"annealed(tau_min)={mean_active:.3f}"
    )
    print("OK: target_k=1 honors ~1 active/row at the annealed (honest) bar.")


def _selftest_softmax_topk_honors_target_k() -> None:
    """Run the full ``softmax_topk`` target_k contract self-test (k>1 and k=1)."""
    _selftest_softmax_topk_multi_atom_honors_target_k()
    _selftest_softmax_topk_k1_converges_to_one_active()


def test_softmax_topk_honors_target_k() -> None:
    _selftest_softmax_topk_honors_target_k()


def test_topk_activation_matches_rust_value_grad() -> None:
    """The ``softmax_topk`` activation is single-sourced on the Rust value+grad.

    ``gamfit.torch``'s ``topk_activation`` calls
    ``gam_sae::assignment::topk_activation_batch_value_grad`` through PyFFI and
    caches the returned diagonal derivative for Torch backward. This pins both
    the marshalled forward value and the autograd bridge.
    """
    from gamfit._binding import rust_module
    from gamfit.torch.penalties import topk_activation

    torch.manual_seed(0)
    tau = 0.37
    logits = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)

    value = topk_activation(logits, tau)
    # A non-uniform upstream so the backward exercises the diagonal Jacobian.
    upstream = torch.arange(1, value.numel() + 1, dtype=torch.float64).reshape_as(value)
    value.backward(upstream)

    rust_value, rust_grad = rust_module().sae_topk_activation_value_grad(
        logits.detach().numpy(), tau
    )
    torch.testing.assert_close(
        value.detach(), torch.from_numpy(rust_value), rtol=0.0, atol=0.0
    )
    # logits.grad = upstream * σ(l/τ); Rust returns the diagonal derivative σ(l/τ).
    expected_grad = upstream * torch.from_numpy(rust_grad)
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, expected_grad, rtol=0.0, atol=0.0)


def test_topk_activation_forward_contains_no_python_numerical_kernel() -> None:
    """Keep #2011's activation numerics in Rust; Python may only bridge autograd."""
    ffi_by_bridge = {
        torch_penalties._JumpReLUSTEFn: "jumprelu_gate_value_grad",
        torch_penalties._IBPMapFn: "sae_ibp_map_batch_value_grad",
        torch_penalties._JumpReLUBoundedGateFn: "sae_jumprelu_batch_value_grad",
        torch_penalties._TopKActivationFn: "sae_topk_activation_value_grad",
    }
    forbidden = ("torch.sigmoid(", "torch.softplus(", "torch.where(", "torch.exp(")
    for bridge, ffi_name in ffi_by_bridge.items():
        source = inspect.getsource(bridge.forward)
        assert ffi_name in source
        assert all(token not in source for token in forbidden)


def test_softmax_topk_numeric_helpers_remain_rust_routed() -> None:
    """Guard #2011's complete named helper inventory at the Python boundary."""
    ffi_by_helper = {
        torch_manifold_sae._duchon_centers_nd: "sae_duchon_centers_nd",
        torch_manifold_sae._SparsityLayer._direction_cluster_anchor: (
            "sae_direction_cluster_anchor"
        ),
        torch_manifold_sae._SparsityLayer._quadratic_subspace_anchor: (
            "sae_quadratic_subspace_anchor"
        ),
        torch_manifold_sae._SparsityLayer._apply_global_anchor_rule: (
            "sae_apply_anchor_rule"
        ),
        torch_manifold_sae._SparsityLayer._matching_pursuit_commit: (
            "sae_matching_pursuit_commit"
        ),
        torch_manifold_sae._SparsityLayer._update_assign_ema: (
            "sae_assign_ema_update"
        ),
        torch_manifold_sae._SparsityLayer._sinkhorn_balance: (
            "sae_sinkhorn_balance_bias"
        ),
    }
    forbidden = (
        "torch.linalg",
        "torch.svd",
        "torch.exp(",
        "torch.log(",
        "torch.logsumexp(",
        "for _ in range(",
    )
    for helper, ffi_name in ffi_by_helper.items():
        source = inspect.getsource(helper)
        assert ffi_name in source
        assert all(token not in source for token in forbidden)

    router = inspect.getsource(
        torch_manifold_sae._SparsityLayer.reconstruction_topk_gate
    )
    assert "_ResidualEmScoreFn.apply" in router
    assert "_sinkhorn_balance" in router
    assert "_matching_pursuit_commit" in router
    assert "_update_assign_ema" in router
    # Softmax, top-k, detaches, and STE arithmetic are deliberately Torch-tape
    # composition; the residual/code criterion itself must not return to Python.
    assert "torch.linalg" not in router
    assert "torch.einsum" not in router


def test_duchon_centers_nd_uses_rust_low_discrepancy_lift() -> None:
    """The product-patch Duchon lift is Rust-owned and preserves the legacy R_d cloud."""
    from gamfit.torch.manifold_sae import _duchon_centers_nd

    centers = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    lifted = _duchon_centers_nd(centers, 3)
    expected = torch.tensor(
        [
            [0.0, 0.2548776662466927, 0.06984029099805333],
            [1.0 / 3.0, 0.009755332493385449, 0.6396805819961064],
            [2.0 / 3.0, 0.764632998740078, 0.20952087299415956],
            [1.0, 0.5195106649867709, 0.7793611639922129],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(lifted, expected, rtol=0.0, atol=2.0e-16)
    assert lifted.dtype == centers.dtype
