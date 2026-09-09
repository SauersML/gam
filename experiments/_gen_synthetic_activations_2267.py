#!/usr/bin/env python
"""Generate a synthetic (N, D) activation matrix at the #2267 documented scale.

The #2267 timing question is a *cost-structure* question: at the shipped
635-row, D=5120 -> PCA-32 shape, does the K={8,32,128,512} x {curved,linear}
ladder finish under 900 s on CPU? Manifold-fit wall time is dominated by the
outer rho-cascade iteration count and the per-eval arrow-Schur solve, both of
which are driven by the (N, P, K) shape and the covariance spectrum -- not by
the literal activation values. A synthetic matrix with a realistic decaying
spectrum plus heavy-tailed idiosyncratic noise reproduces that cost structure
and lets us measure the ladder timing WITHOUT first locating the banked real
activations.npy. If this run finishes comfortably under 900 s, the historical
timeout was the (now-fixed) GPU-death / sparsity-default defect, not a compute
wall; if it does not, we have reproduced the wall at documented scale.

Usage:
  python examples/_gen_synthetic_activations_2267.py --out /scratch.global/<you>/synth_635.npy
  python examples/sae_ev_vs_k_olmo.py --npy /scratch.global/<you>/synth_635.npy \
      --pcs 32 --seed 42 --out /scratch.global/<you>/2267_synth_timing.json
"""
from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=635, help="documented row count")
    ap.add_argument("--dim", type=int, default=5120, help="documented d_model (OLMo-3-32B)")
    ap.add_argument("--rank", type=int, default=48, help="latent rank of the low-dim signal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output .npy path")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # Low-rank signal with a geometrically decaying spectrum (activation-like:
    # a few dominant directions, a long soft tail the PCA-32 keeps most of).
    spectrum = np.geomspace(1.0, 0.02, args.rank)
    loadings = rng.standard_normal((args.rows, args.rank)) * spectrum
    basis, _ = np.linalg.qr(rng.standard_normal((args.dim, args.rank)))
    signal = loadings @ basis.T
    # Heavy-tailed idiosyncratic noise (Student-t) so the retained-variance ratio
    # and the outer-search conditioning resemble real residual streams.
    noise = rng.standard_t(df=4.0, size=(args.rows, args.dim)) * 0.05
    x = (signal + noise).astype(np.float64)
    np.save(args.out, x)
    energy = float(np.sum(x**2))
    print(
        f"wrote {args.out}: shape={x.shape} total_energy={energy:.3e} "
        f"rank={args.rank} seed={args.seed}"
    )


if __name__ == "__main__":
    main()
