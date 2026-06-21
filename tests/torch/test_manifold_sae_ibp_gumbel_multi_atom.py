"""IBP-Gumbel torch ⇄ closed-form integration for ``K>1`` atoms.

Three contracts are pinned here, all for ``K=4``:

1. End-to-end parity: ``ManifoldSAE.fit(x, sparsity={'kind': 'ibp_gumbel', ...})``
   produces the same ``assignments`` (and ``fitted`` / ``reml_score``) as the
   closed-form ``gamfit.sae_manifold_fit(..., assignment='ibp_map', schedule=...)``,
   because both delegate to the same Rust kernel with the same schedule object.

2. Differentiability of the IBP-MAP forward: the torch ``ibp_map`` activation
   carries a Rust value+grad backward; its Jacobian must match a central
   difference of the forward (same numeric-validation technique as
   ``test_manifold_sae_parity.test_isometry_backward_grad_matches_rust_grad_jacobian``).

3. Stick-breaking prior decay: with tied logits the per-atom mass falls off
   geometrically as ``pi_k = (alpha/(alpha+1))^(k+1)`` — the defining property of
   the IBP prior, which a bare ``sigmoid(logits/tau)`` could never produce.

Note: the geometric-decay identity is a property of the *prior map on tied
logits*, not of the *data-fitted* converged assignments (which depend on x),
so it is asserted on the layer forward, not on ``fit.assignments``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

pytest: Any = import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")

import gamfit  # noqa: E402  (after importorskip)


def _make_synth(n: int = 56, d: int = 6, k: int = 4, seed: int = 0) -> np.ndarray:
    """Mixture-of-cosines synthetic with ``k`` latent atoms in ``R^d``."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(n, k))
    bases = rng.standard_normal(size=(k, d))
    bases /= np.linalg.norm(bases, axis=1, keepdims=True) + 1e-9
    amps = rng.uniform(0.5, 1.5, size=(n, k))
    x = (amps * np.cos(angles))[:, :, None] * bases[None, :, :]
    return x.sum(axis=1) + 0.05 * rng.standard_normal((n, d))


def test_ibp_gumbel_fit_k4_matches_closed_form() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    X = _make_synth()
    cfg = gt.ManifoldSAEConfig(
        input_dim=X.shape[1],
        K=4,
        intrinsic_rank=1,
        atom_manifold="circle",
        atom_basis="fourier",
        basis_order=2,
        n_basis_per_atom=4,
        sparsity={"kind": "ibp_gumbel", "init_alpha": 1.2, "tau_start": 1.0, "tau_min": 1.0},
        decoder={"ortho_weight": 0.0, "monotonicity_weight": 0.0},
        reml={"enabled": True, "select": ["lambda"]},
    )

    sae = gt.ManifoldSAE(cfg).double()
    torch_x = torch.as_tensor(X, dtype=torch.float64)
    torch_fit = sae.fit(torch_x, n_iter=10, random_state=42)

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
    )

    np.testing.assert_allclose(
        np.asarray(torch_fit.assignments, dtype=np.float64),
        np.asarray(cf_fit.assignments, dtype=np.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(torch_fit.fitted, cf_fit.fitted, rtol=1e-10, atol=1e-12)
    assert torch_fit.reml_score == pytest.approx(cf_fit.reml_score)
    assert np.asarray(torch_fit.assignments).shape == (X.shape[0], 4)


def test_ibp_gumbel_k4_forward_backward_matches_numeric() -> None:
    from gamfit.torch.penalties import ibp_map  # type: ignore[attr-defined]

    alpha = 1.2
    tau = 0.9
    rng = np.random.default_rng(11)
    logits_np = rng.standard_normal((5, 4))
    logits = torch.as_tensor(logits_np, dtype=torch.float64, requires_grad=True)

    assignments = ibp_map(logits, tau, alpha)
    # Scalarize with a fixed non-uniform weighting so the backprop exercises a
    # full Jacobian-vector product, not just a row sum.
    w_np = rng.standard_normal(assignments.shape)
    w = torch.as_tensor(w_np, dtype=torch.float64)
    (assignments * w).sum().backward()
    assert logits.grad is not None
    autograd_grad = logits.grad.detach().numpy().copy()

    # Central difference of the same scalar objective through the Rust forward.
    h = 1e-6
    numeric = np.zeros_like(logits_np)
    for i in range(logits_np.shape[0]):
        for j in range(logits_np.shape[1]):
            lp = logits_np.copy()
            lm = logits_np.copy()
            lp[i, j] += h
            lm[i, j] -= h
            with torch.no_grad():
                vp = ibp_map(torch.as_tensor(lp, dtype=torch.float64), tau, alpha).numpy()
                vm = ibp_map(torch.as_tensor(lm, dtype=torch.float64), tau, alpha).numpy()
            numeric[i, j] = float(np.sum(w_np * (vp - vm)) / (2.0 * h))

    np.testing.assert_allclose(autograd_grad, numeric, rtol=1e-6, atol=1e-7)


def test_ibp_gumbel_k4_stick_breaking_prior_decays() -> None:
    from gamfit.torch.manifold_sae import _SparsityLayer  # type: ignore[attr-defined]

    alpha = 1.2
    cfg = gt.ManifoldSAEConfig(
        input_dim=4,
        K=4,
        intrinsic_rank=1,
        n_basis_per_atom=4,
        sparsity={"kind": "ibp_gumbel", "init_alpha": alpha, "tau_start": 1.5, "tau_min": 0.5},
    )
    layer = _SparsityLayer(cfg)
    tied = torch.zeros((1, 4), dtype=torch.float64)
    tied_assign = layer(tied)[0].detach().numpy().reshape(-1)
    ratio = alpha / (alpha + 1.0)
    sig0 = float(tied_assign[0])
    for k in range(4):
        np.testing.assert_allclose(tied_assign[k], sig0 * ratio**k, rtol=0.0, atol=1e-8)
    assert np.all(np.diff(tied_assign) < 0.0), "stick-breaking prior must strictly decay"
