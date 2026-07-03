"""Tier-0 shared mean + Tier-1 interference-subspace emitter (#2023 / #2021).

This is the Python-side prototype of the spine's Tier-1 -> Tier-2 hand-off pieces,
mirroring the Rust `gam_sae::tiered` module so both agree numerically:

  * ``tier0_mean``            -- the shared mean mu (Tier-0). Subtract before T1,
                                add back at reconstruct; use as the EV baseline.
  * ``interference_subspace`` -- Tier-1's active subspace ``Q`` (what the linear
                                dictionary explains), its orthogonal complement
                                ``Q_perp``, and the per-direction ``scale``.
  * ``behavioral_fisher_factors`` -- the ready-to-pass GLS weight square-root the
                                curved tier hands to ``sae_manifold_fit`` as
                                ``fisher_factors`` with
                                ``fisher_provenance="behavioral_fisher"`` and
                                ``structured_whitening=False``.

The #2021 coupling: the curved tier's GLS weight ``G = U U^T = I - Q Q^T`` (U spans
``Q_perp``) DOWN-WEIGHTS the directions Tier-1 already explains, so curved atoms
pursue only residual structure. Penalizing ``Q`` itself (the sign error) would make
the curved tier re-chase linear structure -- the whole point is the complement.

Two Tier-1 sources are supported:
  * an atom-lane fit (``gamfit.sparse_dictionary_fit`` -> decoder K x P + sparse
    codes): ``interference_subspace_from_atoms``;
  * a block-lane fit (``gamfit.block_sparse_dictionary_fit`` -> orthonormal block
    frames): ``interference_subspace_from_blocks``.
Both return the same ``InterferenceSubspace``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def tier0_mean(z_train: np.ndarray) -> np.ndarray:
    """The Tier-0 shared mean mu (length P): the train-split column mean.

    Hold this fixed and reuse for out-of-sample de-meaning and the EV baseline so
    held-out EV is measured against the same Tier-0 constant.
    """
    z = np.asarray(z_train, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] == 0:
        raise ValueError(f"tier0_mean needs a non-empty (N, P) matrix; got {z.shape}")
    return z.mean(axis=0)


@dataclass
class InterferenceSubspace:
    """Tier-1's interference model handed to the curved tier.

    ``q``      -- (P, r) orthonormal columns: what Tier-1 explains.
    ``q_perp`` -- (P, P-r) orthonormal columns: the complement (``Q_perp``).
    ``scale``  -- (r,) singular values of the usage-weighted decoder along ``q``.
    """

    q: np.ndarray
    q_perp: np.ndarray
    scale: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.q.shape[1])

    @property
    def ambient_p(self) -> int:
        return int(self.q.shape[0])


def _split_gram(gram: np.ndarray, rank: int | None) -> InterferenceSubspace:
    """Eigendecompose the P x P usage-weighted decoder Gram and split into the
    top-``r`` active subspace ``q`` and the trailing ``q_perp``."""
    gram = 0.5 * (gram + gram.T)  # symmetrize
    evals, evecs = np.linalg.eigh(gram)  # ascending eigenvalues, orthonormal cols
    evals = np.clip(evals, 0.0, None)
    p = gram.shape[0]
    total = float(evals.sum())
    if total <= 0.0:
        raise ValueError("interference_subspace: Tier-1 carries no fired energy")
    if rank is None:
        # smallest r whose top-r energy reaches 99% of the total
        tail = evals[::-1]  # descending
        acc = np.cumsum(tail)
        r = int(np.searchsorted(acc, 0.99 * total) + 1)
        r = max(1, min(p, r))
    else:
        r = max(1, min(p, int(rank)))
    # descending order columns: last is largest
    order = np.argsort(evals)[::-1]
    q_idx = order[:r]
    perp_idx = order[r:]
    q = np.ascontiguousarray(evecs[:, q_idx])
    q_perp = np.ascontiguousarray(evecs[:, perp_idx])
    scale = np.sqrt(evals[q_idx])
    return InterferenceSubspace(q=q, q_perp=q_perp, scale=scale)


def interference_subspace_from_atoms(
    decoder: np.ndarray, indices: np.ndarray, codes: np.ndarray, rank: int | None = None
) -> InterferenceSubspace:
    """From an atom-lane fit: decoder (K, P) + sparse routing (indices/codes,
    N x s). Weight atom k by its fired energy w_k = sum_i codes[i,k]^2, form the
    usage-weighted Gram G = sum_k w_k d_k d_k^T, and split it."""
    decoder = np.asarray(decoder, dtype=np.float64)
    indices = np.asarray(indices)
    codes = np.asarray(codes, dtype=np.float64)
    k, p = decoder.shape
    weight = np.zeros(k, dtype=np.float64)
    np.add.at(weight, indices.reshape(-1), (codes.reshape(-1) ** 2))
    dw = decoder * np.sqrt(weight)[:, None]
    gram = dw.T @ dw
    return _split_gram(gram, rank)


def interference_subspace_from_blocks(t1, rank: int | None = None) -> InterferenceSubspace:
    """From a block-lane fit (``gamfit.BlockSparseDictionaryFit``): the union of
    the orthonormal block frames ``D_g`` (b x P), each weighted by its utilization
    (the fraction of rows that fire block g), forms the usage-weighted Gram."""
    p = int(t1.decoder.shape[1])
    gram = np.zeros((p, p), dtype=np.float64)
    for g in range(int(t1.n_blocks)):
        dg = np.asarray(t1.block_frame(g), dtype=np.float64)  # (b, P), rows orthonormal
        w = float(t1.block_utilization[g])
        if w <= 0.0:
            continue
        gram += w * (dg.T @ dg)
    return _split_gram(gram, rank)


def behavioral_fisher_factors(
    sub: InterferenceSubspace, *, soft_epsilon: float = 0.0, n_rows: int | None = None
) -> np.ndarray:
    """The GLS weight square-root U to pass as ``fisher_factors``.

    ``G = U U^T`` is the per-row GLS weight the curved tier fits under. To make the
    linear dictionary the interference model we DOWN-WEIGHT ``Q`` (penalize the
    complement): the hard form is ``U = Q_perp`` (weight 0 on ``Q``, 1 on
    ``Q_perp`` -> ``G = I - Q Q^T``). ``soft_epsilon > 0`` keeps a small weight on
    ``Q`` for conditioning: ``U = [Q_perp | sqrt(eps) Q]`` -> ``G = I - (1-eps) Q Q^T``.

    Returns U as (P, r_U). If ``n_rows`` is given, broadcast to the (N, P, r_U)
    stack the FFI expects (shared across rows; per-row activity scaling is a
    later refinement owned by the curved tier).
    """
    if soft_epsilon <= 0.0:
        u = sub.q_perp
    else:
        u = np.concatenate([sub.q_perp, np.sqrt(soft_epsilon) * sub.q], axis=1)
    u = np.ascontiguousarray(u, dtype=np.float64)
    if n_rows is None:
        return u
    return np.broadcast_to(u[None, :, :], (int(n_rows), u.shape[0], u.shape[1])).copy()


def _selftest() -> None:
    """Planted-structure check: a 2-D explained subspace + a genuinely
    unexplained direction. Verifies Q/Q_perp orthonormality, that Q_perp captures
    the unexplained direction, and the behavioral-fisher weight sign."""
    rng = np.random.default_rng(0)
    n, p = 2000, 5
    # Tier-1 explains directions e0, e1 strongly, e2 weakly; e3,e4 unexplained.
    basis = np.eye(p)
    codes_true = rng.standard_normal((n, 3)) * np.array([4.0, 2.0, 0.3])
    x = codes_true @ basis[:3]
    # A rank-3 "decoder" reproducing those directions with matching energy.
    decoder = basis[:3].astype(np.float64)
    indices = np.tile(np.arange(3, dtype=np.uint32), (n, 1))
    codes = codes_true.astype(np.float64)

    sub = interference_subspace_from_atoms(decoder, indices, codes, rank=None)
    # Orthonormality.
    assert np.allclose(sub.q.T @ sub.q, np.eye(sub.rank), atol=1e-9), "q not orthonormal"
    assert np.allclose(
        sub.q_perp.T @ sub.q_perp, np.eye(p - sub.rank), atol=1e-9
    ), "q_perp not orthonormal"
    assert np.allclose(sub.q.T @ sub.q_perp, 0.0, atol=1e-9), "q not perp q_perp"
    # The complement must include e3, e4 (unexplained).
    q_perp_span = sub.q_perp @ sub.q_perp.T
    for j in (3, 4):
        proj = q_perp_span[j, j]
        assert proj > 0.99, f"e{j} should live in Q_perp; got projection {proj:.3f}"
    # Behavioral-fisher weight down-weights Q, not Q_perp.
    u = behavioral_fisher_factors(sub, soft_epsilon=0.0)
    g = u @ u.T  # (P,P) = I - Q Q^T
    # Weight on an explained direction (e0) ~ 0; on unexplained (e4) ~ 1.
    assert g[0, 0] < 0.05, f"explained dir e0 should be down-weighted; got {g[0,0]:.3f}"
    assert g[4, 4] > 0.95, f"unexplained dir e4 should keep weight; got {g[4,4]:.3f}"
    # Tier-0 roundtrip.
    mu = tier0_mean(x + 7.0)
    assert np.allclose(mu, x.mean(0) + 7.0, atol=1e-9)
    print(
        f"[tier1_interference] selftest OK: rank(Q)={sub.rank}, scale={np.round(sub.scale,3)}, "
        f"G[e0,e0]={g[0,0]:.3f} (down-weighted), G[e4,e4]={g[4,4]:.3f} (kept)"
    )


if __name__ == "__main__":
    _selftest()
