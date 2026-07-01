#!/usr/bin/env python3
"""Manifold (1D-Duchon) SAE vs traditional linear SAE on held-out reconstruction EV
across dictionary size K (#1026).

Goal: a homogeneous 1D-Duchon manifold dictionary should reconstruct curved-manifold
data better than a pure-linear dictionary at equal K -- and keep beating it as K grows
toward 32_000.

Both sides are Rust-backed (gamfit.sae_ev_vs_k_frontier): the manifold fit uses
`atom_basis = ["duchon"] * K`, `d_atom = 1`; the linear baseline is the Rust sparse
dictionary at the same K. Scored only on a disjoint test split via `predict`.

Run (CPU):
  PYTHONPATH=/Users/user/gam python3.13 /Users/user/gam/bench/manifold_vs_linear_duchon.py \
      --k 16 64 256 --p 48 --n-train 2000 --n-test 800 --true-concepts 12
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from gamfit._sae_manifold import sae_ev_vs_k_frontier


def make_curved_data(
    n: int, p: int, n_concepts: int, n_active: int, harmonics: int, noise: float, seed: int
) -> np.ndarray:
    """Rows are sparse sums of CURVED concept fibers embedded in R^p.

    Each true concept j owns a random band-limited curve gamma_j: [0,1] -> R^p
    (sum of `harmonics` Fourier modes). A row picks `n_active` concepts, each at a
    random coordinate t, and sums amp * gamma_j(t). A linear dictionary must tile
    each curve with many straight atoms; a 1D-Duchon atom can represent the whole
    fiber with one atom -- so the curved dictionary should win.
    """
    rng = np.random.default_rng(seed)
    # Per-concept Fourier coefficients: (n_concepts, 2*harmonics, p)
    coeffs = rng.standard_normal((n_concepts, 2 * harmonics, p)) / np.sqrt(harmonics)

    def gamma(concept: int, t: np.ndarray) -> np.ndarray:
        # t: (m,) -> (m, p)
        feats = []
        for h in range(1, harmonics + 1):
            feats.append(np.cos(2 * np.pi * h * t))
            feats.append(np.sin(2 * np.pi * h * t))
        phi = np.stack(feats, axis=1)  # (m, 2*harmonics)
        return phi @ coeffs[concept]  # (m, p)

    x = np.zeros((n, p), dtype=float)
    for i in range(n):
        active = rng.choice(n_concepts, size=n_active, replace=False)
        ts = rng.uniform(0.0, 1.0, size=n_active)
        amps = rng.uniform(0.5, 1.5, size=n_active)
        for a, (c, t) in enumerate(zip(active, ts)):
            x[i] += amps[a] * gamma(int(c), np.array([t]))[0]
    x += noise * rng.standard_normal((n, p))
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--p", type=int, default=48)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--true-concepts", type=int, default=12)
    ap.add_argument("--n-active", type=int, default=3)
    ap.add_argument("--harmonics", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--n-iter", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--basis",
        default="periodic",
        help="homogeneous atom basis: duchon | periodic | circle | linear. "
        "duchon has a large per-atom border (huge dense evidence cache -> needs "
        "the streaming route even at small K); periodic/circle are small-basis "
        "curved atoms whose dense cache fits at moderate K.",
    )
    args = ap.parse_args()

    print(
        f"data: p={args.p} n_train={args.n_train} n_test={args.n_test} "
        f"true_concepts={args.true_concepts} n_active={args.n_active} "
        f"harmonics={args.harmonics} noise={args.noise}"
    )
    train = make_curved_data(
        args.n_train, args.p, args.true_concepts, args.n_active,
        args.harmonics, args.noise, args.seed + 1,
    )
    test = make_curved_data(
        args.n_test, args.p, args.true_concepts, args.n_active,
        args.harmonics, args.noise, args.seed + 2,
    )

    t0 = time.perf_counter()
    frontier = sae_ev_vs_k_frontier(
        train,
        test,
        k_values=list(args.k),
        hybrid_atom_basis=lambda k: [args.basis] * k,
        d_atom=1,
        sae_fit_kwargs={"n_iter": args.n_iter, "random_state": args.seed},
        linear_fit_kwargs={},
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  K      | manifold_EV | linear_EV | diff      | manifold beats linear?")
    print(f"  -------+-------------+-----------+-----------+-----------------------")
    won = 0
    for row in frontier["rows"]:
        diff = row["hybrid_minus_linear"]
        beat = "YES" if diff > 0 else "no"
        won += diff > 0
        print(
            f"  {row['K']:>6} | {row['hybrid_ev']:>11.4f} | {row['linear_ev']:>9.4f} "
            f"| {diff:>+9.4f} | {beat}"
        )
    print(f"\n  verdict: {frontier['verdict']}")
    print(f"  manifold beat linear at {won}/{len(frontier['rows'])} dictionary sizes")
    print(f"  total wall: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
