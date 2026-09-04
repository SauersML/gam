"""Exact PCA baseline, so the comparator is not under-fitted (issue #2502).

The Rust `pca<m>` arm reaches the top-`m` principal subspace by orthogonal
iteration (one Stiefel block, `x̂ = γ (x D₁ᵀ) D₁`), which converges at rate
`(λ_{m+1}/λ_m)^t`. A baseline that has not converged is a rigged benchmark, so
this script computes the SAME quantity exactly -- eigendecomposition of the
train second moment, held-out FVU at each rank -- and the reported PCA row is
whichever of the two is better.

This is deliberately outside the Rust engine: it is the comparator, and running
it in numpy can only make the dictionary's claim harder.
"""

import argparse
import json
import os

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--ranks", default="16,32")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    train = np.load(os.path.join(args.acts_dir, "train.npy"), mmap_mode="r")
    ev = np.load(os.path.join(args.acts_dir, "eval.npy"), mmap_mode="r")
    n, p = train.shape

    mean = np.zeros(p, dtype=np.float64)
    for s in range(0, n, args.chunk):
        mean += np.asarray(train[s : s + args.chunk], dtype=np.float64).sum(0)
    mean /= n

    cov = np.zeros((p, p), dtype=np.float64)
    for s in range(0, n, args.chunk):
        blk = np.asarray(train[s : s + args.chunk], dtype=np.float64) - mean
        cov += blk.T @ blk
    cov /= n
    w, v = np.linalg.eigh(cov)
    order = np.argsort(-w)
    w, v = w[order], v[:, order]

    m_eval = ev.shape[0]
    eval_mean = np.zeros(p, dtype=np.float64)
    for s in range(0, m_eval, args.chunk):
        eval_mean += np.asarray(ev[s : s + args.chunk], dtype=np.float64).sum(0)
    eval_mean /= m_eval

    ranks = [int(r) for r in args.ranks.split(",")]
    rows = {}
    tss = 0.0
    rss = {r: 0.0 for r in ranks}
    for s in range(0, m_eval, args.chunk):
        blk = np.asarray(ev[s : s + args.chunk], dtype=np.float64)
        centred_train = blk - mean
        centred_eval = blk - eval_mean
        tss += float((centred_eval * centred_eval).sum())
        for r in ranks:
            proj = centred_train @ v[:, :r]
            resid = centred_train - proj @ v[:, :r].T
            rss[r] += float((resid * resid).sum())
    for r in ranks:
        rows[str(r)] = {
            "rank": r,
            "heldout_fvu": rss[r] / tss,
            "heldout_explained_variance": 1.0 - rss[r] / tss,
            "active_scalars_per_token": r,
            "selection_bits_per_token": 0.0,
        }
    report = {
        "source": "numpy eigh of the train second moment (exact PCA comparator)",
        "n_train": int(n),
        "n_eval": int(m_eval),
        "p": int(p),
        "eigenvalue_mass_top16": float(w[:16].sum() / w.sum()),
        "eigenvalue_mass_top32": float(w[:32].sum() / w.sum()),
        "ranks": rows,
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
