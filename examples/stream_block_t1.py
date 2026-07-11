"""Streaming block-sparse Tier-1 over a sharded corpus -> seeds manifest.

The node run's verbatim T1 driver: point it at a ``residual_shard`` harvest
directory (the format ``examples/residual_shard_io.py`` writes, e.g. a 30M-row
Qwen3 bf16 harvest) and it streams a block-sparse Tier-1 fit shard-by-shard —
never materialising the corpus — then emits the block->chart **seeds manifest**
(``<out>.seeds.json`` + ``<out>.seeds.npz``) a K=1-per-block Tier-2 nursery (or
``compose_tiers.py --from-seeds``) consumes.

Pipeline::

    shards-dir --(ShardReader.batches)--> BlockSparseDictStream
        fit_begin(seed) -> [partial_fit(shard) ... end_epoch()] x epochs
        -> finalize() -> to_fit(sample) -> seed_manifest -> JSON + npz

All heavy state (block frames, gamma, per-block cross-moments, revival reservoir)
lives native-side; each shard round-trips only its own ``batch x d_model`` rows.

Usage::

    python stream_block_t1.py --shards-dir /path/to/harvest --n-blocks 4096 \
        --block-size 2 --block-topk 1 --aux-k 64 --out /path/to/seeds/qwen_L20
    python stream_block_t1.py --synthetic --n-blocks 6 --out /tmp/smoke   # local
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, Iterator

import numpy as np

import gamfit

# examples/ is not a package; import the sibling manifest emitter by path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compose_tiers import emit_block_seed_manifest  # noqa: E402


def _synthetic_corpus(n: int, p: int, n_blocks: int, block_size: int, seed: int) -> np.ndarray:
    """A planted product-of-block-subspaces corpus: each row lives in ONE of
    ``n_blocks`` orthonormal ``block_size``-planes (plus light noise), so a streamed
    block-sparse T1 has genuine block structure to recover."""
    rng = np.random.default_rng(seed)
    # Orthonormal planted atoms (columns of a random orthogonal basis).
    q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    atoms = q[:, : n_blocks * block_size].T  # (K, p), orthonormal rows
    x = np.zeros((n, p), dtype=np.float32)
    for i in range(n):
        t = i % n_blocks
        coeffs = rng.standard_normal(block_size).astype(np.float32) + 0.5
        acc = np.zeros(p, dtype=np.float32)
        for r in range(block_size):
            acc += coeffs[r] * atoms[t * block_size + r]
        x[i] = acc
    x += 0.03 * rng.standard_normal((n, p)).astype(np.float32)
    return np.ascontiguousarray(x)


def _batch_source(
    args: argparse.Namespace,
) -> tuple[Callable[[], Iterator[np.ndarray]], np.ndarray, int]:
    """Return ``(make_batches, seed_sample, d_model)``.

    ``make_batches()`` yields fresh ``(<=batch, d_model)`` float32 blocks for one
    epoch (re-callable per epoch). ``seed_sample`` is a bounded representative slice
    used to seed the frames AND to build the manifest.
    """
    if args.shards_dir is not None:
        from residual_shard_io import load_shards

        reader = load_shards(args.shards_dir)
        d_model = int(reader.d_model)

        def make_batches() -> Iterator[np.ndarray]:
            return reader.batches(args.batch)

        # Seed / manifest sample: the first `seed_rows` rows.
        parts: list[np.ndarray] = []
        got = 0
        for batch in reader.batches(args.batch):
            take = min(args.seed_rows - got, batch.shape[0])
            parts.append(np.ascontiguousarray(batch[:take]))
            got += take
            if got >= args.seed_rows:
                break
        seed_sample = np.concatenate(parts, axis=0) if parts else np.empty((0, d_model), np.float32)
        return make_batches, seed_sample, d_model

    # Synthetic in-memory corpus streamed in row-batches.
    x = _synthetic_corpus(args.n_tokens, args.p, args.n_blocks, args.block_size, args.random_state)
    d_model = x.shape[1]

    def make_batches() -> Iterator[np.ndarray]:
        for start in range(0, x.shape[0], args.batch):
            yield x[start : start + args.batch]

    seed_sample = x[: min(args.seed_rows, x.shape[0])]
    return make_batches, seed_sample, d_model


def stream_block_t1(args: argparse.Namespace) -> dict:
    make_batches, seed_sample, d_model = _batch_source(args)
    if seed_sample.shape[0] == 0:
        raise SystemExit("no rows found to seed the fit (empty corpus?)")
    print(
        f"[stream_block_t1] d_model={d_model}, seed_rows={seed_sample.shape[0]}, "
        f"G={args.n_blocks} b={args.block_size} top-k={args.block_topk} aux_k={args.aux_k}"
    )

    stream = gamfit.block_sparse_dictionary_fit_begin(
        seed_sample,
        args.n_blocks,
        block_size=args.block_size,
        block_topk=args.block_topk,
        max_epochs=args.max_epochs,
        minibatch=args.minibatch,
        block_tile=args.block_tile,
        frame_ridge=args.frame_ridge,
        aux_k=args.aux_k,
        tolerance=args.tolerance,
    )

    for epoch in range(args.max_epochs):
        rows = 0
        for shard in make_batches():
            stats = stream.partial_fit(shard)
            rows += stats["rows"]
        ep = stream.end_epoch()
        print(
            f"[epoch {ep['epoch']:>3}] rows={rows} EV={ep['explained_variance']:.4f} "
            f"gamma={ep['gamma']:.3f} dead={ep['dead']} "
            f"accepted_births={ep['accepted_births']} "
            f"converged={ep['converged']}"
        )
        if ep["converged"]:
            break

    art = stream.finalize()
    print(
        f"[stream_block_t1] finalized: EV={art.explained_variance:.4f} "
        f"epochs={art.epochs} converged={art.converged}"
    )

    # Route the sample back through the frozen frames to recover the FULL block->
    # chart seeding surface, then emit the manifest a Tier-2 nursery consumes.
    fit = art.to_fit(seed_sample)
    manifest = emit_block_seed_manifest(
        fit, seed_sample, args.out, n_basis_chart=args.n_basis_chart
    )
    live = int(np.sum(np.asarray(fit.block_utilization) > 0.0))
    print(
        f"[stream_block_t1] manifest: {manifest['n_blocks']} blocks "
        f"({live} live), b={manifest['block_size']}; "
        + (f"written to {args.out}.seeds.json(+.npz)" if args.out else "not written (--out unset)")
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_argument_group("data source")
    src.add_argument("--shards-dir", default=None,
                     help="residual_shard harvest directory (real activations)")
    src.add_argument("--synthetic", action="store_true",
                     help="use a planted in-memory corpus (default when no --shards-dir)")
    src.add_argument("--n-tokens", type=int, default=4000, help="synthetic corpus rows")
    src.add_argument("--p", type=int, default=32, help="synthetic ambient dim")
    src.add_argument("--seed-rows", type=int, default=8192,
                     help="rows used to seed frames + build the manifest")

    blk = ap.add_argument_group("block Tier-1")
    blk.add_argument("--n-blocks", type=int, default=6, help="number of blocks G (K=G*b)")
    blk.add_argument("--block-size", type=int, default=2, help="atoms per block b (2-4)")
    blk.add_argument("--block-topk", type=int, default=1, help="blocks active per row")
    blk.add_argument("--aux-k", type=int, default=2, help="AuxK dead-block revival budget")
    blk.add_argument("--max-epochs", type=int, default=30)
    blk.add_argument("--batch", type=int, default=65536, help="rows per streamed shard/batch")
    blk.add_argument("--minibatch", type=int, default=512)
    blk.add_argument("--block-tile", type=int, default=1024)
    blk.add_argument("--frame-ridge", type=float, default=1.0e-9)
    blk.add_argument("--tolerance", type=float, default=1.0e-6)
    blk.add_argument("--n-basis-chart", type=int, default=4)

    ap.add_argument("--out", default=None,
                    help="seeds manifest path prefix (<out>.seeds.json + .npz)")
    ap.add_argument("--random-state", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    return stream_block_t1(args)


if __name__ == "__main__":
    main()
