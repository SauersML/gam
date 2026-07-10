#!/usr/bin/env python3
"""Qwen3.6-35B concept arena: manifold SAE vs flat TopK on REAL L17
residual-stream harvests (weekday / month / color prompt banks), scored by the
same un-Goodhartable axes as bench/real_llm_arena.py: held-out reconstruction
R² at matched L0 and a ridge probe to the REAL concept label (3-way), with the
raw-PCA skyline. Weekday/month rows additionally carry known circular
structure — the reason these banks exist — so the discovered per-atom
topologies are reported for inspection.

Inputs are the dose-harvest caches ``harvest_cache_<concept>_L17_n<k>.npz``
(key ``X_last`` = (n, 2048) last-token activations).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from real_llm_arena import _pca_fit, _r2, _ridge_probe
from bsf_manifold_zoo import _fit_flat_topk, _fit_ours_rust


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True,
                        help="Directory holding harvest_cache_*_L17_*.npz")
    parser.add_argument("--pca", type=int, default=64)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--atoms", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--rust-iters", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--featurizers", default="ours_rust,flat")
    parser.add_argument("--out", default="qwen_concept_arena.jsonl")
    args = parser.parse_args()

    xs, labels = [], []
    for path in sorted(Path(args.cache_dir).glob("harvest_cache_*_L17_*.npz")):
        concept = path.name.split("_")[2]
        x = np.asarray(np.load(path)["X_last"], dtype=np.float64)
        xs.append(x)
        labels.extend([concept] * x.shape[0])
    if not xs:
        raise SystemExit(f"no harvest caches under {args.cache_dir}")
    acts = np.concatenate(xs, axis=0)
    y = np.asarray(labels)
    n = acts.shape[0]
    print(f"loaded {n} rows x {acts.shape[1]} from {len(xs)} concept banks")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    n_hold = max(1, int(round(args.holdout_fraction * n)))
    hold, fit = order[:n_hold], order[n_hold:]

    mu, comps = _pca_fit(acts[fit], min(args.pca, fit.size - 1))
    proj = (acts - mu[None, :]) @ comps.T
    scale = proj[fit].std(axis=0).clip(min=1e-12)
    proj = proj / scale[None, :]

    out_path = Path(args.out)
    skyline = _ridge_probe(proj[fit], y[fit], proj[hold], y[hold])
    header = {"record": "config", "n_rows": n, "pca": int(comps.shape[0]),
              "atoms": args.atoms, "top_k": args.top_k, "seed": args.seed,
              "probe_skyline_concept": skyline}
    with out_path.open("a") as fh:
        fh.write(json.dumps(header) + "\n")
    print(f"[skyline pca] concept probe {skyline:.3f}")

    for which in [w.strip() for w in args.featurizers.split(",") if w.strip()]:
        t0 = time.perf_counter()
        if which == "flat":
            fitted = _fit_flat_topk(proj[fit], proj, width=4 * args.atoms,
                                    k=args.top_k, steps=args.steps,
                                    batch_size=128, lr=1e-3, seed=args.seed,
                                    device="cpu")
        elif which == "ours_rust":
            fitted = _fit_ours_rust(proj[fit], proj, atoms=args.atoms,
                                    top_k=args.top_k, n_iter=args.rust_iters,
                                    seed=args.seed)
        else:
            raise SystemExit(f"unknown featurizer {which!r}")
        codes = np.abs(np.asarray(fitted.gate, dtype=np.float64))
        l0 = float(np.mean(np.sum(codes > 1e-8, axis=1)))
        record = {
            "record": "result", "featurizer": fitted.name,
            "heldout_recon_r2": _r2(proj[hold], fitted.recon[hold]),
            "mean_l0": l0,
            "probe_acc_concept": _ridge_probe(codes[fit], y[fit], codes[hold], y[hold]),
            "fit_seconds": fitted.fit_seconds,
            "wall_seconds": time.perf_counter() - t0,
        }
        with out_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"[{fitted.name}] heldout R2 {record['heldout_recon_r2']:.4f} "
              f"L0 {l0:.2f} concept probe {record['probe_acc_concept']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
