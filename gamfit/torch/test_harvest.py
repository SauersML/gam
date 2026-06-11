"""Synthetic-model test for the output-Fisher harvest contract.

A fixed linear head ``logits = x @ Wᵀ`` (so ``J_n = W`` *exactly*, identical for
every token regardless of ``x``) gives an analytically known pullback
``G_n = Wᵀ F_n W`` with ``F_n = diag(p_n) − p_n p_nᵀ`` the softmax Fisher at
``p_n = softmax(W x_n)``. We dense-eigendecompose ``G_n`` as the closed-form
reference (the *reference* may form matrices; the harvest path may not) and check:

* the harvested factors ``U_n`` reconstruct the rank-r truncation of ``G_n``:
  ``U_n U_nᵀ ≈ Σ_{k≤r} λ_k v_k v_kᵀ`` — this is sign/rotation invariant, the
  exact gauge-freedom the contract tolerates;
* the top-r Ritz values match the closed-form top-r eigenvalues;
* ``mass_residual`` matches the analytic tail ``Σ_{k>r} λ_k``;
* the round-tripped ``.npz`` shard has the exact ``(X, U, mass_residual)`` shapes
  / dtypes ``RowMetric::OutputFisher`` consumes.

Fixed seeds throughout; no clock entropy.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gamfit.torch.harvest import (  # noqa: E402
    HarvestShard,
    harvest_downstream_output_fisher_factors,
    harvest_output_fisher_factors,
    load_harvest_shard,
    save_harvest_shard,
)


class _LinearHead(torch.nn.Module):
    """``logits = x @ Wᵀ`` with the hook site = the identity-passed input ``x``.

    ``feature`` is an ``nn.Identity`` whose output is the hook-site activation
    ``x_n``; ``head`` is a fixed linear map, so ``∂logits/∂x_n = W`` for every
    token, independent of ``x``. That makes the pullback ``G_n = Wᵀ F_n W``
    exactly computable in closed form.
    """

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.feature = torch.nn.Identity()
        self.head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            self.head.weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature(x))


def _closed_form_pullback(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``G = Wᵀ F W`` with ``F = diag(p) − p pᵀ``, ``p = softmax(W x)`` — dense."""
    logits = W @ x
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    F = np.diag(p) - np.outer(p, p)
    return W.T @ F @ W


def test_harvest_matches_closed_form_linear_head() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    C, p, n = 6, 5, 4  # classes, activation dim, tokens
    rank = 2

    W_np = rng.standard_normal((C, p)).astype(np.float64)
    X_np = rng.standard_normal((n, p)).astype(np.float64)

    W = torch.from_numpy(W_np).to(torch.float64)
    X = torch.from_numpy(X_np).to(torch.float64)

    model = _LinearHead(W).to(torch.float64)

    shard = harvest_output_fisher_factors(
        model,
        model.feature,
        X,
        rank=rank,
        oversample=3,
        n_iter=4,
        trace_probes=p,  # exhaustive-ish trace probes on a tiny p
        seed=0,
    )

    assert isinstance(shard, HarvestShard)
    assert shard.X.shape == (n, p)
    assert shard.U.shape == (n, p, rank)
    assert shard.mass_residual.shape == (n,)

    # X must be the hook-site activation (here the identity-passed input).
    np.testing.assert_allclose(shard.X, X_np.astype(np.float32), rtol=0, atol=1e-5)

    max_factor_err = 0.0
    max_eig_err = 0.0
    max_residual_err = 0.0

    for i in range(n):
        G = _closed_form_pullback(W_np, X_np[i])
        evals, evecs = np.linalg.eigh(G)  # ascending
        order = np.argsort(evals)[::-1]
        evals = np.clip(evals[order], 0.0, None)
        evecs = evecs[:, order]

        # Closed-form rank-r reconstruction W_r = Σ_{k<r} λ_k v_k v_kᵀ.
        top = evecs[:, :rank] * np.sqrt(evals[:rank])[None, :]
        W_r_ref = top @ top.T

        U_n = shard.U[i].astype(np.float64)  # (p, r)
        W_r_got = U_n @ U_n.T

        # Sign/rotation-invariant: compare the reconstructed rank-r operators.
        max_factor_err = max(max_factor_err, float(np.max(np.abs(W_r_got - W_r_ref))))

        # Top-r eigenvalues recovered (Σ over captured factors = Σ λ_k via
        # trace of the reconstruction).
        got_eigsum = float(np.trace(W_r_got))
        ref_eigsum = float(evals[:rank].sum())
        max_eig_err = max(max_eig_err, abs(got_eigsum - ref_eigsum))

        # mass_residual = analytic tail Σ_{k>=r} λ_k.
        ref_residual = float(evals[rank:].sum())
        max_residual_err = max(
            max_residual_err, abs(float(shard.mass_residual[i]) - ref_residual)
        )

    assert max_factor_err < 1e-4, f"factor recon error {max_factor_err}"
    assert max_eig_err < 1e-4, f"eigenvalue error {max_eig_err}"
    assert max_residual_err < 1e-3, f"mass_residual error {max_residual_err}"


def test_shard_roundtrip_schema(tmp_path) -> None:
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    C, p, n = 4, 3, 2
    rank = 2
    W = torch.from_numpy(rng.standard_normal((C, p))).to(torch.float64)
    X = torch.from_numpy(rng.standard_normal((n, p))).to(torch.float64)
    model = _LinearHead(W).to(torch.float64)

    shard = harvest_output_fisher_factors(
        model, model.feature, X, rank=rank, seed=0
    )

    out = save_harvest_shard(shard, tmp_path / "shard")
    assert out.endswith(".npz")

    loaded = load_harvest_shard(out)
    # f64 promotion at the gam boundary.
    assert loaded["X"].dtype == np.float64
    assert loaded["U"].dtype == np.float64
    assert loaded["mass_residual"].dtype == np.float64
    assert loaded["X"].shape == (n, p)
    assert loaded["U"].shape == (n, p, rank)
    assert loaded["mass_residual"].shape == (n,)
    assert loaded["rank"] == rank

    # Row-major U flatten must equal the RowMetric layout u[n, i*r + k].
    flat = loaded["U"].reshape(n, p * rank)
    for i in range(n):
        for a in range(p):
            for k in range(rank):
                assert flat[i, a * rank + k] == loaded["U"][i, a, k]

    # Default provenance is the same-position metric, and it round-trips.
    assert shard.provenance == "output_fisher"
    assert loaded["provenance"] == "output_fisher"


class _CausalSumHead(torch.nn.Module):
    """``logits_t = W · (Σ_{s ≤ t} x_s)`` — a prefix-sum over positions then a
    fixed linear head.

    This is the minimal model with KV-path-like forward influence: the
    activation at position ``n`` feeds *every* future position ``t ≥ n`` with
    ``∂logits_t/∂x_n = W`` (and zero for ``t < n``, the causal mask). The
    same-position pullback at ``n`` sees only ``t = n``; the downstream pullback
    aggregates all ``t ≥ n``, so the downstream Fisher mass is strictly larger
    for early positions.
    """

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.feature = torch.nn.Identity()
        self.head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            self.head.weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hook site = the raw per-position activation x_n (feature is identity),
        # then a causal prefix sum mixes positions, then the linear head. So
        # ∂logits_t/∂x_n = W for t ≥ n and 0 otherwise.
        h = self.feature(x)
        prefix = torch.cumsum(h, dim=0)  # (n_pos, p)
        return self.head(prefix)


def test_downstream_equals_same_position_for_position_local_model(tmp_path) -> None:
    # A plain linear head has ∂logits_t/∂x_n = 0 for t ≠ n, so the downstream
    # aggregate over future positions collapses to the same-position pullback.
    # The two harvests must therefore produce identical factors (up to the
    # rank-r sign/rotation gauge) — only the provenance tag differs.
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    C, p, n, rank = 5, 4, 3, 2
    W = torch.from_numpy(rng.standard_normal((C, p))).to(torch.float64)
    X = torch.from_numpy(rng.standard_normal((n, p))).to(torch.float64)
    model = _LinearHead(W).to(torch.float64)

    same = harvest_output_fisher_factors(model, model.feature, X, rank=rank, seed=0)
    down = harvest_downstream_output_fisher_factors(
        model, model.feature, X, rank=rank, seed=0
    )

    assert same.provenance == "output_fisher"
    assert down.provenance == "output_fisher_downstream"
    # Compare the rank-r reconstructions U_n U_nᵀ (gauge-invariant).
    for i in range(n):
        a = same.U[i].astype(np.float64)
        b = down.U[i].astype(np.float64)
        np.testing.assert_allclose(a @ a.T, b @ b.T, rtol=0, atol=1e-4)

    # The downstream provenance round-trips through the shard I/O.
    out = save_harvest_shard(down, tmp_path / "down")
    loaded = load_harvest_shard(out)
    assert loaded["provenance"] == "output_fisher_downstream"


def test_downstream_aggregates_future_positions() -> None:
    # With the causal-sum head, ∂logits_t/∂x_n = W for every t ≥ n, so the
    # downstream pullback at an early position aggregates more future positions
    # than the same-position one — strictly more Fisher mass. The very last
    # position has no future, so the two coincide there.
    torch.manual_seed(5)
    rng = np.random.default_rng(5)
    C, p, n, rank = 5, 4, 4, 3
    W = torch.from_numpy(rng.standard_normal((C, p))).to(torch.float64)
    X = torch.from_numpy(rng.standard_normal((n, p))).to(torch.float64)
    model = _CausalSumHead(W).to(torch.float64)

    same = harvest_output_fisher_factors(
        model, model.feature, X, rank=rank, trace_probes=p, seed=0
    )
    down = harvest_downstream_output_fisher_factors(
        model, model.feature, X, rank=rank, trace_probes=p, seed=0
    )

    # Total captured Fisher mass per row = trace(U_n U_nᵀ) + mass_residual.
    def total_mass(shard, i):
        u = shard.U[i].astype(np.float64)
        return float(np.trace(u @ u.T)) + float(shard.mass_residual[i])

    # Early positions: strictly more downstream mass (more future positions).
    assert total_mass(down, 0) > total_mass(same, 0) + 1e-6
    # Last position: no future ⇒ downstream coincides with same-position.
    np.testing.assert_allclose(
        total_mass(down, n - 1), total_mass(same, n - 1), rtol=0, atol=1e-4
    )
