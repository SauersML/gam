"""Streaming Arrow-Schur SAE-manifold joint fit at LLM-atom scale.

This drives the *real* ``gamfit.sae_manifold_fit`` joint solve — not a
synthetic stand-in. The fit dispatches its execution plan from the problem
size and the hardware memory budget (``GpuRuntime``): when the dense in-core
working set ``n_obs · (M_total + K) · 8 bytes`` exceeds the in-core threshold
(a fraction of the device budget, or a conservative host-RAM fraction on a
CPU-only box), the Rust core streams the fit in row chunks instead of
materializing the full ``(N × M_total)`` basis / ``(N × K)`` logit buffers.

The chunk size and the stream-vs-in-core decision are auto-derived by the same
``rust_module().sae_streaming_plan`` the fit consults internally, so this demo
also reports the plan it expects before running. Euclidean atoms at large ``K``
exercise the block-sparse atom Schur path (cost scales with ``k_active`` per
token, not ``K``); the streaming driver accumulates the reduced ``β``-system
online and, when a CUDA device is present, runs the dominant ``O(K²)`` reduced
solve on-device.

The Python process only ever holds ``Z`` of shape ``(N, P)`` — the per-atom
basis, the per-row latent blocks, and the reduced Schur system are built and
discarded chunk-by-chunk inside Rust, which is what lets the co-fit run as an
LLM-scale teacher without materializing billions of activation rows.
"""

from __future__ import annotations

import numpy as np

import gamfit
from gamfit._binding import rust_module


# Atom count in the LLM-SAE regime. With Euclidean ``d_atom`` atoms the dense
# in-core working set crosses the streaming threshold well before this scale,
# so the joint fit auto-dispatches to the chunked path. K is kept large enough
# to exercise the block-sparse atom Schur accumulation while the demo still
# finishes on a CPU-only box.
N_ATOMS = 4_096
N_OBS = 24_000
N_FEATURES = 32
D_ATOM = 2
MAX_ITER = 3


def _synthetic_low_rank(n: int, p: int, latent_dim: int, seed: int) -> np.ndarray:
    """Centered low-rank-plus-noise response the SAE atoms can reconstruct."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n, latent_dim))
    mixing = rng.standard_normal((latent_dim, p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    z = latent @ mixing + 0.05 * rng.standard_normal((n, p))
    z -= z.mean(axis=0, keepdims=True)
    return np.ascontiguousarray(z)


def main() -> None:
    z = _synthetic_low_rank(N_OBS, N_FEATURES, D_ATOM, seed=0)

    # Euclidean atoms use a monomial patch basis; query the same dispatcher the
    # fit consults so we can report the plan it will follow. `total_basis` is
    # the summed per-atom basis size; the dispatcher only needs its scale to
    # size chunks, so we pass the conservative monomial-patch estimate.
    per_atom_basis = 1 + D_ATOM + (D_ATOM * (D_ATOM + 1)) // 2  # degree-2 patch
    total_basis = N_ATOMS * per_atom_basis
    plan = dict(rust_module().sae_streaming_plan(
        N_OBS, total_basis, N_ATOMS, D_ATOM, total_basis * N_FEATURES
    ))
    use_streaming, chunk_size = bool(plan["streaming"]), int(plan["chunk_size"])

    fit = gamfit.sae_manifold_fit(
        X=z,
        K=N_ATOMS,
        atom_topology="euclidean",
        d_atom=D_ATOM,
        assignment="topk",
        top_k=8,
        n_iter=MAX_ITER,
        random_state=0,
    )

    plan = "streaming" if use_streaming else "in-core"
    print(
        f"sae_manifold_fit K={N_ATOMS:,} n_obs={N_OBS:,} "
        f"plan={plan} chunk_size={chunk_size:,} "
        f"r2={fit.reconstruction_r2:.4f}"
    )


if __name__ == "__main__":
    main()
