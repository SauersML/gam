"""Regression for issue #356 — ``ManifoldSAE.fit()`` joint solve across configs.

Before the fix (gamfit 0.1.134) the closed-form joint Arrow-Schur co-fit
panicked or raised across every non-trivial config:

============== =============== ============= ================================
atom_manifold  intrinsic_rank  sparsity      pre-fix failure
============== =============== ============= ================================
circle         1               ibp_gumbel    PanicException broadcast
                                              ``[N*D, M] -> [N, M]``
circle         1               softmax_topk  same broadcast panic
sphere         2               jumprelu      same broadcast panic
product        2               ibp_gumbel    ValueError ``sae_build_duchon_atom:
                                              primary penalty was not built``
============== =============== ============= ================================

Root cause (commit fc6b16088):

* The first three rows panicked in the SAE pre-fit decoder identifiability
  audit. A single ``p``-vector predictor per row has decoder Jacobian
  ``(n*p, M_k*p)``, which the cross-block flat/channel-aware audit conflated
  with the ``(n)``-row placeholder design and broadcast ``(n*p, ...)`` into
  ``(n, ...)``. The audit now runs a per-atom rank check on the weighted design
  ``D_k = diag(a_.k)·Φ_k`` directly (the ``p``-fold output replication carries
  no extra structural information), so the ``(n*p, ...)`` block is never
  materialised.

* The ``product`` row failed earlier, in the Duchon atom penalty assembly: the
  scale-free pure-Duchon path emits the curvature seminorm as
  ``OperatorStiffness`` (never ``Primary``), but ``sae_build_duchon_atom``
  searched for ``Primary``. It now selects ``OperatorStiffness`` and builds the
  penalty exactly like ``duchon_function_norm_penalty`` (power 0).

This test pins the headline capability: ``.fit()`` must run the joint solve to
completion and return finite decoder blocks / reconstruction / assignments for
every row of the matrix. The issue explicitly noted that no torch test exercised
``.fit()``, which is why the regression shipped silently.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

pytest: Any = import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")


def _make_synth(n: int = 48, d: int = 12, k: int = 4, seed: int = 0) -> np.ndarray:
    """Mixture-of-cosines synthetic with ``k`` latent atoms in ``R^d``."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(n, k))
    bases = rng.standard_normal(size=(k, d))
    bases /= np.linalg.norm(bases, axis=1, keepdims=True) + 1e-9
    amps = rng.uniform(0.5, 1.5, size=(n, k))
    x = (amps * np.cos(angles))[:, :, None] * bases[None, :, :]
    return x.sum(axis=1) + 0.05 * rng.standard_normal((n, d))


# (atom_manifold, intrinsic_rank, sparsity-kind) — exactly the #356 matrix.
_CONFIG_MATRIX = [
    ("circle", 1, {"kind": "ibp_gumbel", "target_k": 3}),
    ("circle", 1, {"kind": "softmax_topk", "target_k": 3}),
    ("sphere", 2, {"kind": "jumprelu"}),
    ("product", 2, {"kind": "ibp_gumbel", "target_k": 3}),
]


@pytest.mark.parametrize("atom_manifold,intrinsic_rank,sparsity", _CONFIG_MATRIX)
def test_fit_joint_solve_does_not_panic_across_configs(
    atom_manifold: str, intrinsic_rank: int, sparsity: dict[str, Any]
) -> None:
    # N=48, D=12, n_atoms=8, n_basis_per_atom=6 — the issue repro dimensions.
    torch.manual_seed(0)
    np.random.seed(0)
    X = _make_synth(n=48, d=12)
    cfg = gt.ManifoldSAEConfig(
        input_dim=12,
        n_atoms=8,
        n_basis_per_atom=6,
        intrinsic_rank=intrinsic_rank,
        atom_manifold=atom_manifold,
        sparsity=sparsity,
        dtype=torch.float64,
    )
    sae = gt.ManifoldSAE(cfg)

    # Pre-fix this either panicked in Rust (PanicException, uncatchable as a
    # normal exception path) or raised ValueError. It must now complete.
    fit = sae.fit(torch.as_tensor(X, dtype=torch.float64), max_iter=3, random_state=0)

    blocks = np.asarray(fit.decoder_blocks, dtype=np.float64)
    assert np.isfinite(blocks).all(), "decoder blocks must be finite after the joint solve"
    fitted = np.asarray(fit.fitted, dtype=np.float64)
    assert fitted.shape == (48, 12)
    assert np.isfinite(fitted).all(), "reconstruction must be finite"
    assignments = np.asarray(fit.assignments, dtype=np.float64)
    assert assignments.shape == (48, 8)
    assert np.isfinite(assignments).all(), "assignments must be finite"
