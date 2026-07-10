#!/usr/bin/env python3
"""Train and package the fixed-64 block candidate for issue #2251.

The production math stays in Rust through
``gamfit.fixed_budget_block_sparse_dictionary_fit``.  This script only loads an
independent SAE training corpus, routes the frozen emotion-task tokens, performs
the task's fixed mean pooling, and appends the resulting arm to the audit archive
consumed by ``fixed_budget_probe_2251.py``.

The input audit ``.npz`` must already contain ``labels``, ``train_indices``,
``test_indices``, ``document_offsets``, ``token_activations``, the raw skyline,
and the scalar TopK-64 baseline.  ``--dictionary-train`` is an independent
``N_train x P`` activation matrix; task labels never enter dictionary fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ACTIVE = 64


def mean_pool_sparse_block_route(
    blocks: np.ndarray,
    codes: np.ndarray,
    document_offsets: np.ndarray,
    n_atoms: int,
    block_size: int,
) -> np.ndarray:
    """Mean-pool sparse signed block coordinates into document features."""
    if blocks.ndim != 2 or codes.shape != (*blocks.shape, block_size):
        raise ValueError(
            f"blocks/codes must have shapes (T,k)/(T,k,{block_size}); got {blocks.shape}/{codes.shape}"
        )
    n_documents = document_offsets.size - 1
    lengths = np.diff(document_offsets)
    if document_offsets[0] != 0 or document_offsets[-1] != blocks.shape[0] or np.any(lengths <= 0):
        raise ValueError("every document must own a non-empty contiguous token interval")
    document = np.repeat(np.arange(n_documents, dtype=np.int64), lengths)
    feature = (
        blocks[:, :, None].astype(np.int64) * block_size
        + np.arange(block_size, dtype=np.int64)[None, None, :]
    )
    if feature.size and (feature.min() < 0 or feature.max() >= n_atoms):
        raise ValueError("block route exceeds the scalar dictionary capacity")
    doc_grid = np.broadcast_to(document[:, None, None], feature.shape)
    pooled = np.zeros((n_documents, n_atoms), dtype=np.float32)
    np.add.at(pooled, (doc_grid.reshape(-1), feature.reshape(-1)), codes.reshape(-1))
    pooled /= lengths[:, None]
    return pooled


def _archive_dict(archive: Any) -> dict[str, np.ndarray]:
    return {key: np.asarray(archive[key]) for key in archive.files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-input", type=Path, required=True)
    parser.add_argument("--dictionary-train", type=Path, required=True)
    parser.add_argument("--n-atoms", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--name", default="block_fixed64")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--block-tile", type=int, default=1024)
    parser.add_argument("--frame-ridge", type=float, default=1.0e-9)
    parser.add_argument("--aux-k", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--route-chunk", type=int, default=8192)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    args = parser.parse_args()
    if args.block_size <= 1:
        raise ValueError("the anti-competition candidate requires block_size > 1")
    if args.route_chunk <= 0:
        raise ValueError("route_chunk must be positive")

    source = np.load(args.audit_input, allow_pickle=False)
    payload = _archive_dict(source)
    for key in ("labels", "document_offsets", "token_activations", "reps__raw"):
        if key not in payload:
            raise ValueError(f"audit input is missing {key!r}")
    tokens = np.asarray(payload["token_activations"], dtype=np.float32)
    offsets = np.asarray(payload["document_offsets"], dtype=np.int64)
    train = np.load(args.dictionary_train, mmap_mode="r")
    if train.ndim != 2 or tokens.ndim != 2 or train.shape[1] != tokens.shape[1]:
        raise ValueError(
            f"dictionary/task activations must be matrices with the same P; got {train.shape}/{tokens.shape}"
        )
    if not np.isfinite(tokens).all():
        raise ValueError("task token activations contain a non-finite value")

    import gamfit

    fit = gamfit.fixed_budget_block_sparse_dictionary_fit(
        np.ascontiguousarray(train, dtype=np.float32),
        args.n_atoms,
        active=ACTIVE,
        block_size=args.block_size,
        max_epochs=args.max_epochs,
        minibatch=args.minibatch,
        block_tile=args.block_tile,
        frame_ridge=args.frame_ridge,
        aux_k=args.aux_k,
        tolerance=args.tolerance,
    )
    if fit.n_atoms != args.n_atoms or fit.active != ACTIVE:
        raise RuntimeError(
            f"fixed-budget fit returned K={fit.n_atoms}, active={fit.active}; expected {args.n_atoms}/{ACTIVE}"
        )

    block_parts, gate_parts, code_parts = [], [], []
    for start in range(0, tokens.shape[0], args.route_chunk):
        end = min(start + args.route_chunk, tokens.shape[0])
        blocks, gates, codes = fit.transform(tokens[start:end])
        block_parts.append(blocks)
        gate_parts.append(gates)
        code_parts.append(codes)
    blocks = np.ascontiguousarray(np.concatenate(block_parts), dtype=np.uint32)
    gates = np.ascontiguousarray(np.concatenate(gate_parts), dtype=np.float32)
    codes = np.ascontiguousarray(np.concatenate(code_parts), dtype=np.float32)
    pooled = mean_pool_sparse_block_route(
        blocks, codes, offsets, fit.n_atoms, fit.block_size
    )

    name = args.name
    for key in (
        f"reps__{name}",
        f"active__{name}",
        f"n_atoms__{name}",
        f"block_size__{name}",
        f"route_indices__{name}",
        f"route_values__{name}",
    ):
        if key in payload:
            raise ValueError(f"refusing to overwrite existing audit arm key {key!r}")
    payload[f"reps__{name}"] = pooled
    payload[f"active__{name}"] = np.asarray([ACTIVE], dtype=np.int64)
    payload[f"n_atoms__{name}"] = np.asarray([fit.n_atoms], dtype=np.int64)
    payload[f"block_size__{name}"] = np.asarray([fit.block_size], dtype=np.int64)
    payload[f"route_indices__{name}"] = blocks
    payload[f"route_values__{name}"] = codes
    payload[f"route_gates__{name}"] = gates

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.model_out,
        decoder=np.asarray(fit.decoder, dtype=np.float32),
        gamma=np.asarray([fit.gamma], dtype=np.float32),
        block_size=np.asarray([fit.block_size], dtype=np.int64),
        block_topk=np.asarray([fit.block_topk], dtype=np.int64),
        n_atoms=np.asarray([fit.n_atoms], dtype=np.int64),
        active=np.asarray([fit.active], dtype=np.int64),
        block_utilization=np.asarray(fit.block_utilization, dtype=np.float32),
        block_stable_rank=np.asarray(fit.block_stable_rank, dtype=np.float32),
    )
    print(
        json.dumps(
            {
                "arm": name,
                "n_atoms": fit.n_atoms,
                "active": fit.active,
                "block_size": fit.block_size,
                "block_topk": fit.block_topk,
                "explained_variance": fit.explained_variance,
                "epochs": fit.epochs,
                "audit_out": str(args.out),
                "model_out": str(args.model_out),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
