#!/usr/bin/env python3
"""Per-arm :class:`FittedFeaturizer` builders for the #1026 Eq-4 bits scoring.

Each #1026 arm reconstructs the test cloud with a different dictionary; gam#2233
scores them all on the SAME Eq-4 description-length currency. To do that every
arm must present the uniform scoring surface the zoo uses
(``bench/bsf_manifold_zoo.FittedFeaturizer``): a per-atom firing ``gate`` (N,G),
a per-atom SOLO contribution callable, ``code_dims`` (1 for a flat latent, d+1
for a curved chart: intrinsic coords + amplitude), the one-time
``dictionary_params`` cost, and the overall ``recon``.

The three wired arms (gam#2233 task 3): ``external_topk``, ``gam_flat``, and
``hybrid_rust``. The hybrid arm stacks a flat block (``code_dims`` 1) beside curved atoms (``code_dims``
1+d) so the theorem's support/code/residual decomposition is visible per tier.

Memory note (K up to 32768): the scorer only ever reads the gate through
``gate > 1e-10`` and indexes ``atom_contribution(g)[take]`` with ``take`` a small
row subset. We therefore (a) pass a single nonnegative magnitude gate — the code
SPECTRUM is a per-row-sign-invariant SVD, so the |code| magnitude reproduces the
exact singular values while the arm's own signed ``recon`` carries the residual —
and (b) return a lazy row-indexable contribution proxy so no full (N, P) atom
array is ever materialized.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable

import numpy as np

from bits_eq4 import FittedFeaturizer


def _ensure_bench_on_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    bench_dir = os.path.join(repo_root, "bench")
    for cand in (bench_dir, repo_root):
        if cand not in sys.path and os.path.isdir(cand):
            sys.path.insert(0, cand)


class _RowLazyContribution:
    """Row-indexable proxy: ``proxy[take]`` computes only the requested rows.

    Satisfies the scorer's sole access pattern ``atom_contribution(g)[take]``
    without allocating the full (N, P) solo reconstruction of an atom.
    """

    __slots__ = ("_fn",)

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self._fn = fn

    def __getitem__(self, idx: Any) -> np.ndarray:
        return self._fn(np.asarray(idx))


def _dense_mag_gate(indices: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    """Scatter sparse (indices, codes) into a dense nonnegative (N, K) gate."""
    n = indices.shape[0]
    gate = np.zeros((n, k), dtype=np.float32)
    rows = np.arange(n)[:, None]
    gate[rows, indices.astype(np.int64)] = np.abs(codes).astype(np.float32)
    return gate


# --------------------------------------------------------------------------- #
# Flat blocks (external TopK torch bar, gam sparse-dictionary lane)            #
# --------------------------------------------------------------------------- #
def _flat_block_from_sparse(
    x_bits: np.ndarray, decoder: np.ndarray, indices: np.ndarray, codes: np.ndarray,
    recon: np.ndarray,
):
    """Common flat-block scoring surface from a sparse (indices, codes) routing.

    Returns (gate (N,K), atom_contribution(g)->lazy, code_dims (K,), dict_params).
    """
    k = decoder.shape[0]
    gate = _dense_mag_gate(indices, codes, k)

    def atom_contribution(g: int):
        col = gate[:, g]  # nonnegative magnitude; SVD spectrum is sign-invariant
        dec_g = decoder[g]
        return _RowLazyContribution(lambda take, col=col, dec_g=dec_g: np.outer(col[take], dec_g))

    return gate, atom_contribution, np.ones(k, dtype=int), int(decoder.size)


def build_external_topk(x_bits, *, W_enc, W_dec, b_dec, top_k) -> FittedFeaturizer:
    """FittedFeaturizer for the Gao TopK bar, re-encoding x_bits in numpy."""
    pre = (x_bits.astype(np.float32) - b_dec[None, :]) @ W_enc.T  # (N, K)
    k = W_dec.shape[0]
    top_k = min(int(top_k), k)
    topi = np.argpartition(-pre, top_k - 1, axis=1)[:, :top_k]  # (N, top_k)
    rows = np.arange(pre.shape[0])[:, None]
    topv = np.maximum(pre[rows, topi], 0.0)  # ReLU (Gao TopK)
    gate, contrib, code_dims, dparams = _flat_block_from_sparse(
        x_bits, W_dec, topi, topv, recon=None)
    recon = np.einsum("nk,nkp->np", topv, W_dec[topi]) + b_dec[None, :]
    return FittedFeaturizer(
        name="external_topk", gate=gate, atom_contribution=contrib,
        code_dims=code_dims, dictionary_params=dparams,
        recon=recon.astype(np.float64), fit_seconds=0.0)


def build_gam_flat(x_bits, *, fit) -> FittedFeaturizer:
    """FittedFeaturizer for gamfit.sparse_dictionary_fit on x_bits."""
    tr = fit.transform(x_bits)
    recon = fit.reconstruct(tr.indices, tr.codes)
    gate, contrib, code_dims, dparams = _flat_block_from_sparse(
        x_bits, np.asarray(fit.decoder), tr.indices, tr.codes, recon=recon)
    return FittedFeaturizer(
        name="gam_flat", gate=gate, atom_contribution=contrib,
        code_dims=code_dims, dictionary_params=dparams,
        recon=np.asarray(recon, dtype=np.float64), fit_seconds=0.0)


# --------------------------------------------------------------------------- #
# Curved blocks (native sae_manifold tier)                                    #
# --------------------------------------------------------------------------- #
def _curved_block_rust(r_bits, *, model):
    """Curved tier from a fitted gamfit.sae_manifold model on residual r_bits."""
    _ensure_bench_on_path()
    from synth_sae_bench_manifold import _basis_values

    payload = model.converged_latents(r_bits)
    assignments = np.asarray(payload["assignments"], dtype=float)  # (N, K)
    recon = np.asarray(payload["fitted"], dtype=float)             # (N, P)
    coords = [np.asarray(c, dtype=float) for c in payload["coords"]]
    blocks = [np.asarray(b, dtype=float) for b in model.decoder_blocks]

    def atom_contribution(g: int):
        basis = model.basis_specs[g]
        n_harm = model._n_harmonics[g] if g < len(model._n_harmonics) else 1
        centers = model._duchon_centers[g] if g < len(model._duchon_centers) else None

        def fn(take, g=g, basis=basis, n_harm=n_harm, centers=centers):
            phi = _basis_values(basis, coords[g][take], n_harm, centers)
            rows = min(phi.shape[1], blocks[g].shape[0])
            return assignments[take, g:g + 1] * (phi[:, :rows] @ blocks[g][:rows])

        return _RowLazyContribution(fn)

    d_atoms = [c.shape[1] if c.ndim == 2 else 1 for c in coords]
    gate = np.abs(assignments)
    code_dims = np.asarray([d + 1 for d in d_atoms], dtype=int)
    return gate, atom_contribution, code_dims, int(sum(b.size for b in blocks)), recon


def _stack_blocks(flat, curved, recon_full) -> FittedFeaturizer:
    """Concatenate a flat block and a curved block into one FittedFeaturizer.

    ``flat`` = (gate_f, contrib_f, code_dims_f, dparams_f); ``curved`` =
    (gate_c, contrib_c, code_dims_c, dparams_c, _recon_c). Atom index g < K_flat
    routes to the flat contribution; g >= K_flat to the curved one.
    """
    gate_f, contrib_f, cd_f, dp_f = flat
    gate_c, contrib_c, cd_c, dp_c, _ = curved
    k_flat = gate_f.shape[1]
    gate = np.concatenate([gate_f, gate_c.astype(np.float32)], axis=1)
    code_dims = np.concatenate([cd_f, cd_c])

    def atom_contribution(g: int):
        return contrib_f(g) if g < k_flat else contrib_c(g - k_flat)

    return FittedFeaturizer(
        name="hybrid", gate=gate, atom_contribution=atom_contribution,
        code_dims=code_dims, dictionary_params=dp_f + dp_c,
        recon=np.asarray(recon_full, dtype=np.float64), fit_seconds=0.0)


def build_hybrid_rust(
    x_bits, r_bits, *, flat_fit, curved_model, recon_full,
) -> FittedFeaturizer:
    """FittedFeaturizer for the all-Rust hybrid: sparse-dict flat + Rust curved."""
    tr = flat_fit.transform(x_bits)
    flat = _flat_block_from_sparse(
        x_bits, np.asarray(flat_fit.decoder), tr.indices, tr.codes, recon=None)[:4]
    curved = _curved_block_rust(r_bits, model=curved_model)
    fit = _stack_blocks(flat, curved, recon_full)
    fit.name = "hybrid_rust"
    return fit
