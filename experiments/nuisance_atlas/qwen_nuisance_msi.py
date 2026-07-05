#!/usr/bin/env python3
"""Positional nuisance-atlas on the Qwen3-8B wikitext harvest (RUN ON MSI).

Tests whether the massive-activation channel that dominates certain layers
(L18: 99% of variance in one PC) is a POSITIONAL / attention-sink artifact.

The harvest (`harvest_acts.py`) wrote `resid_L{L}.npy` as the real (non-pad)
residual rows, concatenated over docs in corpus order, each doc contributing its
truncated `L_doc` tokens in position order (max_length=512). Doc lengths VARY, so
per-token position is NOT a fixed period of the row index — it is reconstructed
here by re-tokenizing the identical corpus (tokenizer only, no model, fully
deterministic) and validated against the harvest's token-count checksum.

For each layer we report the centred R² a positional design absorbs:
  * fourier   -- a normalized-position Fourier series cos/sin(2π j p/Pmax), j=1..H
  * pos0      -- a single first-token indicator (the attention-sink test)
  * early     -- indicators for the first few positions
  * combined  -- all of the above
and a PERMUTED-POSITION NULL (positions shuffled) for `combined`: if the signal
is genuinely positional the null collapses to the ~M/N overfit floor, ruling out
"any low-rank design fits a near-rank-1 matrix".

Usage (on MSI):
  python qwen_nuisance_msi.py \
    --harvest /projects/standard/hsiehph/sauer354/harvest_out/qwen3_8b_wikitext \
    --model Qwen/Qwen3-8B --layers 18,30,6 --out /tmp/qwen_nuisance.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

PMAX = 512  # harvest seq_len / truncation length


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def reconstruct_corpus(n_docs: int, dataset: str, cfg: str) -> list[str]:
    """Reproduce harvest_acts.py's corpus exactly: first n_docs streamed docs with
    stripped length > 200."""
    from datasets import load_dataset

    ds = load_dataset(dataset, cfg, split="train", streaming=True)
    texts: list[str] = []
    for row in ds:
        t = row.get("text", "") or row.get("content", "")
        if t and len(t.strip()) > 200:
            texts.append(t.strip())
        if len(texts) >= n_docs:
            break
    return texts


def reconstruct_positions(tok, texts: list[str], n_rows: int) -> np.ndarray:
    """Per-row within-doc position, matching the harvest's row order: for each doc
    in order, arange(L_doc) with L_doc the truncated (max_length=PMAX) token count,
    concatenated and capped at n_rows."""
    pos_chunks: list[np.ndarray] = []
    total = 0
    for t in texts:
        ids = tok(t, truncation=True, max_length=PMAX)["input_ids"]
        ld = len(ids)
        pos_chunks.append(np.arange(ld, dtype=np.int32))
        total += ld
    positions = np.concatenate(pos_chunks)
    log(f"reconstructed positions: {positions.shape[0]} real tokens over {len(texts)} docs "
        f"(harvest processed ~301226; rows in npy = {n_rows})")
    return positions[:n_rows], total


def normalized_fourier(positions: np.ndarray, harmonics: int) -> np.ndarray:
    u = positions.astype(np.float64) / PMAX
    cols = []
    for j in range(1, harmonics + 1):
        cols.append(np.cos(2.0 * np.pi * j * u))
        cols.append(np.sin(2.0 * np.pi * j * u))
    return np.stack(cols, axis=1) if cols else np.zeros((positions.shape[0], 0))


def early_indicators(positions: np.ndarray, k: int) -> np.ndarray:
    return np.stack([(positions == q).astype(np.float64) for q in range(k)], axis=1)


def accumulate(npy_path: str, designs: list[np.ndarray], batch_rows: int = 20000):
    """One streaming pass over the memmapped bank for a LIST of designs (each N×M):
    per design G=ZᵀZ, C=ZᵀX; shared Σx, Σx², n. Reading the 4.9 GB bank once for
    all designs (super + permuted null) halves the I/O."""
    mm = np.load(npy_path, mmap_mode="r")
    n, d = mm.shape
    gs = [np.zeros((z.shape[1], z.shape[1])) for z in designs]
    cs = [np.zeros((z.shape[1], d)) for z in designs]
    sx = np.zeros(d)
    sx2 = np.zeros(d)
    for i in range(0, n, batch_rows):
        x = np.asarray(mm[i : i + batch_rows], dtype=np.float64)
        for k, z in enumerate(designs):
            zb = z[i : i + x.shape[0]]
            gs[k] += zb.T @ zb
            cs[k] += zb.T @ x
        sx += x.sum(0)
        sx2 += (x * x).sum(0)
    return [(gs[k], cs[k], sx, sx2, n) for k in range(len(designs))]


def absorbed(acc, cols: list[int]) -> float:
    """Centred aggregate R² of the sub-design on `cols` (must include intercept)."""
    g, c, sx, sx2, n = acc
    gg = g[np.ix_(cols, cols)]
    cc = c[cols, :]
    b = np.linalg.pinv(gg) @ cc
    ss_res = sx2 - 2.0 * np.einsum("kj,kj->j", b, cc) + np.einsum("kj,kj->j", b, gg @ b)
    ss_tot = sx2 - sx * sx / n
    tot = float(ss_tot.sum())
    return 0.0 if tot <= 0.0 else 1.0 - float(ss_res.sum()) / tot


def build_super_design(positions: np.ndarray, harmonics: int, early_k: int):
    """[intercept | fourier(2H) | early(early_k)] plus the column-index map so the
    fourier-only / pos0-only / early-only sub-designs are slices of one accumulator."""
    n = positions.shape[0]
    intercept = np.ones((n, 1))
    fourier = normalized_fourier(positions, harmonics)
    early = early_indicators(positions, early_k)
    super_z = np.concatenate([intercept, fourier, early], axis=1)
    nf = fourier.shape[1]
    ne = early.shape[1]
    cols = {
        "fourier": [0] + list(range(1, 1 + nf)),
        "pos0": [0, 1 + nf],  # early indicator q=0 is the first-token sink
        "early": [0] + list(range(1 + nf, 1 + nf + ne)),
        "combined": list(range(1 + nf + ne)),
    }
    return super_z, cols


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harvest", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--layers", default="18,30,6")
    ap.add_argument("--harmonics", type=int, default=16)
    ap.add_argument("--early-k", type=int, default=8)
    ap.add_argument("--n-docs", type=int, default=6000)
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    ap.add_argument("--out", default="/tmp/qwen_nuisance.json")
    ap.add_argument(
        "--positions",
        default="",
        help="load precomputed per-row positions .npy (skips corpus/tokenizer — for "
        "compute nodes without network); if empty, reconstruct from the corpus",
    )
    ap.add_argument(
        "--save-positions",
        default="",
        help="reconstruct positions, save to this .npy, and exit (run on a login node "
        "with network)",
    )
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.harvest, "manifest.json")))
    layers = [int(x) for x in args.layers.split(",")]
    # Row count from the first layer's bank.
    first_file = os.path.join(args.harvest, f"resid_L{layers[0]}.npy")
    n_rows = int(np.load(first_file, mmap_mode="r").shape[0])

    if args.positions:
        positions = np.load(args.positions).astype(np.int32)
        total_tokens = int(positions.shape[0])
        log(f"loaded {total_tokens} precomputed positions from {args.positions}")
        if positions.shape[0] < n_rows:
            raise SystemExit(
                f"precomputed positions ({positions.shape[0]}) shorter than rows ({n_rows})"
            )
        positions = positions[:n_rows]
    else:
        log("loading tokenizer", args.model)
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        log("reconstructing corpus")
        texts = reconstruct_corpus(args.n_docs, args.dataset, args.dataset_config)
        log(f"corpus docs: {len(texts)}")
        positions, total_tokens = reconstruct_positions(tok, texts, n_rows)
        if args.save_positions:
            np.save(args.save_positions, positions.astype(np.int32))
            log(f"saved {positions.shape[0]} positions to {args.save_positions}; exiting")
            return
        if positions.shape[0] != n_rows:
            raise SystemExit(
                f"position/row misalignment: {positions.shape[0]} positions vs {n_rows} rows "
                f"(reconstructed {total_tokens} tokens; check tokenizer/corpus reproduction)"
            )

    super_z, cols = build_super_design(positions, args.harmonics, args.early_k)
    rng = np.random.default_rng(0)
    null_z = super_z[rng.permutation(n_rows)]  # positions shuffled vs activations

    results = {
        "model": manifest.get("model"),
        "seq_len": manifest.get("seq_len"),
        "n_rows": n_rows,
        "reconstructed_tokens": int(total_tokens),
        "harmonics": args.harmonics,
        "early_k": args.early_k,
        "n_design_combined": super_z.shape[1],
        "layers": {},
    }
    for L in layers:
        path = os.path.join(args.harvest, f"resid_L{L}.npy")
        if not os.path.exists(path):
            log(f"L{L}: {path} missing, skipping")
            continue
        stat = manifest["layers"].get(str(L), {})
        row = {
            "ev_top1": stat.get("ev_top1"),
            "participation_ratio": stat.get("participation_ratio"),
        }
        acc, null_acc = accumulate(path, [super_z, null_z])  # single pass
        for name, cc in cols.items():
            row[f"absorbed_{name}"] = absorbed(acc, cc)
            log(f"L{L} {name}: absorbed {row[f'absorbed_{name}']:.4f}")
        row["absorbed_combined_null"] = absorbed(null_acc, cols["combined"])
        log(f"L{L} combined_null: absorbed {row['absorbed_combined_null']:.4f}")
        results["layers"][str(L)] = row

    print("RESULTS_JSON " + json.dumps(results))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    log("wrote", args.out)


if __name__ == "__main__":
    main()
