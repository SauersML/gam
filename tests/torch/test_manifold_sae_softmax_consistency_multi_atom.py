"""Softmax (soft-assignment) torch ⇄ closed-form parity for ``K>1`` atoms.

``ManifoldSAE.fit(x, sparsity={'kind': 'softmax_topk', ...})`` delegates to the
same Rust ``sae_manifold_fit_minimal`` kernel as ``gamfit.sae_manifold_fit(...,
assignment='softmax')``, with the same temperature schedule object. The
existing parity test (``test_manifold_sae_parity``) only covers the
``ibp_gumbel`` arm. This module pins the structurally-equivalent ``softmax``
arm for ``K>1`` end-to-end: identical ``assignments``, ``fitted`` and
``reml_score``.

Temperature is held constant (``tau_start == tau_min``) so there is no
annealing drift to reason about — both the torch ``fit`` and the closed-form
call build the *same* :class:`GumbelTemperatureSchedule` from the same config
and feed it to the same kernel, so parity is by construction.
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
    """Mixture-of-cosines synthetic with ``k`` latent atoms in ``R^d``."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(n, k))
    bases = rng.standard_normal(size=(k, d))
    bases /= np.linalg.norm(bases, axis=1, keepdims=True) + 1e-9
    amps = rng.uniform(0.5, 1.5, size=(n, k))
    x = (amps * np.cos(angles))[:, :, None] * bases[None, :, :]
    return x.sum(axis=1) + 0.05 * rng.standard_normal((n, d))


def test_softmax_fit_matches_closed_form_k3() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    X = _make_synth()
    # Constant temperature: tau_start == tau_min => the linear schedule never
    # anneals, so the kernel sees a fixed tau on every iteration.
    cfg = gt.ManifoldSAEConfig(
        input_dim=X.shape[1],
        K=3,
        intrinsic_rank=1,
        atom_manifold="circle",
        atom_basis="fourier",
        basis_order=2,
        n_basis_per_atom=4,
        sparsity={"kind": "softmax_topk", "tau_start": 0.5, "tau_min": 0.5},
        decoder={"ortho_weight": 0.0, "monotonicity_weight": 0.0},
        reml={"enabled": True, "select": ["lambda"]},
    )
    assert cfg.closed_form_assignment() == "softmax"

    sae = gt.ManifoldSAE(cfg).double()
    torch_x = torch.as_tensor(X, dtype=torch.float64)
    initializers = sae._closed_form_initializers(torch_x)
    torch_fit = sae.fit(torch_x, n_iter=10, random_state=42)

    # Equivalent closed-form call: same Rust kernel, same amortized warm starts
    # and same (constant) schedule.
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

    np.testing.assert_allclose(
        np.asarray(torch_fit.assignments, dtype=np.float64),
        np.asarray(cf_fit.assignments, dtype=np.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(torch_fit.fitted, cf_fit.fitted, rtol=1e-10, atol=1e-12)
    assert torch_fit.reml_score == pytest.approx(cf_fit.reml_score)
    for b_torch, b_cf in zip(torch_fit.decoder_blocks, cf_fit.decoder_blocks):
        np.testing.assert_allclose(b_torch, b_cf, rtol=1e-10, atol=1e-12)


def test_softmax_module_forward_matches_fit_k3() -> None:
    # After a softmax K=3 fit, the module forward reproduces the closed-form
    # reconstruction (it routes through _forward_from_closed_form).
    torch.manual_seed(3)
    np.random.seed(3)
    X = _make_synth(seed=3)
    cfg = gt.ManifoldSAEConfig(
        input_dim=X.shape[1],
        K=3,
        intrinsic_rank=1,
        atom_manifold="circle",
        atom_basis="fourier",
        n_basis_per_atom=4,
        sparsity={"kind": "softmax_topk", "tau_start": 0.5, "tau_min": 0.5},
    )
    sae = gt.ManifoldSAE(cfg).double()
    x = torch.as_tensor(X, dtype=torch.float64)
    fit = sae.fit(x, n_iter=10, random_state=42)
    out = sae(x)
    fitted = torch.as_tensor(np.asarray(fit.fitted, dtype=np.float64), dtype=torch.float64)
    assert out.assignments.shape == (x.shape[0], 3)
    assert torch.allclose(out.x_hat, fitted, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        out.assignments.detach().numpy(),
        np.asarray(fit.assignments, dtype=np.float64),
        rtol=1e-10,
        atol=1e-12,
    )
