"""Tier-0 shared mean + Tier-1 interference-subspace emitter (#2023 / #2021).

This is the Python-side prototype of the spine's Tier-1 -> Tier-2 hand-off pieces,
mirroring the Rust `gam_sae::tiered` module so both agree numerically:

  * ``tier0_mean``            -- the shared mean mu (Tier-0). Subtract before T1,
                                add back at reconstruct; use as the EV baseline.
  * ``interference_subspace`` -- Tier-1's active subspace ``Q`` (what the linear
                                dictionary explains), its orthogonal complement
                                ``Q_perp``, and the per-direction ``scale``.
                                DIAGNOSTIC ONLY (span reporting).

RETRACTED (audit 2026-07-03): using ``Q_perp`` as a GLS weight ``G = I - Q Q^T`` on
the Tier-2 residual is a PROVEN DESIGN ERROR. A curve's chords span the curve's OWN
plane, so the post-linear curvature signal (chord-sag) lives INSIDE ``span(Q)`` --
measured counterexample: 95.4% of the residual energy inside Q, and the Q_perp
weight crushed the in-plane signal RMS 0.176 -> 0.0001. Curvature is a constraint
AMONG the directions Tier-1 spans, not a missed direction. DOCTRINE: Tier-2 fits the
RAW residual; anti-rechasing is priced by the evidence criterion (never a projector);
the only sanctioned reweighting is a soft Sigma_hat estimated from the actual
(anisotropic) residual. ``demonstrate_qperp_blindness`` below pins the failure.

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


def demonstrate_qperp_blindness(
    sub: InterferenceSubspace, residual: np.ndarray
) -> tuple[float, float]:
    """RETRACTED-WEIGHT GUARDRAIL: shows why ``G = I - Q Q^T`` must NOT be used.

    Applies the (retracted) hard ``Q_perp`` GLS weight to ``residual`` and returns
    ``(rms_before, rms_after)``. When the residual is in-plane curvature (lives in
    ``span(Q)`` -- which it does, because chords span the curve's own plane), the
    weight crushes it to ~0: it is BLIND to exactly what the curved tier exists to
    model. Kept only to pin the failure; do NOT install this as a metric. Tier-2
    fits the RAW residual; anti-rechasing is the criterion's job, not a projector.
    """
    r = np.asarray(residual, dtype=np.float64)
    rms_before = float(np.sqrt(np.mean(r * r)))
    if r.ndim == 2:
        kept = (sub.q_perp @ (sub.q_perp.T @ r.T)).T  # keep only the complement
    else:
        kept = sub.q_perp @ (sub.q_perp.T @ r)
    rms_after = float(np.sqrt(np.mean(kept * kept)))
    return rms_before, rms_after


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
    # The complement must include e3, e4 (unexplained) -- span diagnostic only.
    q_perp_span = sub.q_perp @ sub.q_perp.T
    for j in (3, 4):
        proj = q_perp_span[j, j]
        assert proj > 0.99, f"e{j} should live in Q_perp; got projection {proj:.3f}"
    # BLINDNESS DEMO (retraction guardrail): an in-plane curvature signal lives in
    # span(Q); the retracted Q_perp weight crushes it to ~noise -- proving it is
    # blind to exactly what the curved tier must model.
    in_plane = np.zeros(p)
    in_plane[0], in_plane[1] = 0.6, -0.8  # unit vector inside span(Q) = {e0, e1}
    rms_before, rms_after = demonstrate_qperp_blindness(sub, in_plane)
    assert rms_before > 0.1, f"planted in-plane signal should be nonzero; got {rms_before}"
    assert rms_after < 1e-6, f"Q_perp weight must crush the in-plane signal; got {rms_after}"
    # Tier-0 roundtrip.
    mu = tier0_mean(x + 7.0)
    assert np.allclose(mu, x.mean(0) + 7.0, atol=1e-9)
    print(
        f"[tier1_interference] selftest OK: rank(Q)={sub.rank}, scale={np.round(sub.scale,3)}; "
        f"Q_perp blindness: in-plane RMS {rms_before:.3f} -> {rms_after:.2e} (crushed) "
        f"=> DO NOT use Q_perp as a weight; fit the raw residual"
    )


if __name__ == "__main__":
    _selftest()
