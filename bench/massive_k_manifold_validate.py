#!/usr/bin/env python3
"""Massive-K manifold SAE end-to-end validation harness (#1026).

Drives ``gamfit.sae_manifold_fit`` directly (NOT the stale ``sae_ev_vs_k_frontier``
helper) at a sweep of dictionary sizes K to verify the whole pipeline RUNS at
scale and to capture the wall-time scaling curve.

Design choices per the massive-K streaming contract:
  * ``assignment="jumprelu"`` -> canonicalizes to the ``threshold_gate`` family,
    which is per-row independent and therefore streams cleanly across chunks at
    large K. (``ordered_beta_bernoulli`` couples rows via a cross-row Woodbury mass and refuses
    multi-chunk streaming, blowing up the dense working set at large K.)
  * ``d_atom=1`` intrinsic coordinate per atom (1D manifold fibers).
  * homogeneous ``atom_basis`` / ``atom_topology`` so every atom is the same
    curved primitive.

Reports, per K: RUN ok?, wall seconds, in-sample reconstruction_r2, peak RSS.

Run (CPU):
  PYTHONPATH=/Users/user/gam python3.13 bench/massive_k_manifold_validate.py \
      --k 256 1024 4096 16000 32000 --p 48 --n-train 40000
"""
from __future__ import annotations

import argparse
import resource
import time
import traceback

import numpy as np


def make_curved_data(
    n: int, p: int, n_concepts: int, n_active: int, harmonics: int, noise: float, seed: int
) -> np.ndarray:
    """Rows are sparse sums of CURVED concept fibers embedded in R^p."""
    rng = np.random.default_rng(seed)
    coeffs = rng.standard_normal((n_concepts, 2 * harmonics, p)) / np.sqrt(harmonics)

    def gamma(concept: int, t: np.ndarray) -> np.ndarray:
        feats = []
        for h in range(1, harmonics + 1):
            feats.append(np.cos(2 * np.pi * h * t))
            feats.append(np.sin(2 * np.pi * h * t))
        phi = np.stack(feats, axis=1)
        return phi @ coeffs[concept]

    x = np.zeros((n, p), dtype=float)
    # vectorized concept fiber accumulation
    for i in range(n):
        active = rng.choice(n_concepts, size=n_active, replace=False)
        ts = rng.uniform(0.0, 1.0, size=n_active)
        amps = rng.uniform(0.5, 1.5, size=n_active)
        for a, (c, t) in enumerate(zip(active, ts)):
            x[i] += amps[a] * gamma(int(c), np.array([t]))[0]
    x += noise * rng.standard_normal((n, p))
    return x


def peak_rss_gb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    import sys

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    div = 1024**3 if sys.platform == "darwin" else 1024**2
    return r / div


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, nargs="+", default=[256, 1024, 4096, 16000, 32000])
    ap.add_argument("--p", type=int, default=48)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--true-concepts", type=int, default=24)
    ap.add_argument("--n-active", type=int, default=3)
    ap.add_argument("--harmonics", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--assignment", default="jumprelu")
    ap.add_argument("--basis", default="circle",
                    help="homogeneous atom_topology: circle | periodic | duchon | linear")
    ap.add_argument("--d-atom", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=None,
                    help="cap per-row active atoms so the reconstruction Tower4 primaries stay <=16")
    # Convergence hyperparameters. The harness previously fell back to the facade
    # defaults (isometry=1.0, ard=on, sparsity=1.0, smoothness=1.0), which are ~100x
    # the values the KNOWN-GOOD tests converge under and drive dictionary co-collapse.
    # Default here to the known-good regime so a fit actually converges.
    ap.add_argument("--sparsity-weight", type=float, default=0.01)
    ap.add_argument("--smoothness-weight", type=float, default=0.01)
    ap.add_argument("--isometry-weight", type=float, default=0.0)
    ap.add_argument("--learning-rate", type=float, default=1.0)
    ap.add_argument("--ard", action="store_true", help="enable per-atom ARD (off by default: known-good regime)")
    ap.add_argument("--allow-overcomplete", action="store_true",
                    help="permit N < K (overcomplete dictionary) — keeps the dense "
                         "n×K assignment logits small enough to fit a RAM-tight box")
    args = ap.parse_args()

    import gamfit

    print(f"# massive-K manifold SAE e2e validation")
    print(f"# assignment={args.assignment} basis={args.basis} d_atom={args.d_atom} "
          f"n_iter={args.n_iter}")
    print(f"# data: p={args.p} n_train={args.n_train} true_concepts={args.true_concepts} "
          f"n_active={args.n_active} harmonics={args.harmonics} noise={args.noise}")

    max_k = max(args.k)
    n_train = args.n_train
    if n_train <= max_k and not args.allow_overcomplete:
        n_train = max_k + max(1000, max_k // 4)
        print(f"# bumped n_train to {n_train} to satisfy N > K at K={max_k}")
    elif n_train <= max_k:
        print(f"# OVERCOMPLETE: N={n_train} < K={max_k} (overcomplete dictionary; the "
              f"dense n×K logits stay {n_train*max_k*8/1e9:.2f} GB — fits a RAM-tight box)")

    t0 = time.perf_counter()
    train = make_curved_data(
        n_train, args.p, args.true_concepts, args.n_active,
        args.harmonics, args.noise, args.seed + 1,
    )
    print(f"# built train {train.shape} in {time.perf_counter()-t0:.1f}s\n")

    print(f"{'K':>7} | {'run':>4} | {'wall_s':>9} | {'recon_r2':>9} | {'peak_rss_gb':>11} | note")
    print(f"{'-'*7}-+-{'-'*4}-+-{'-'*9}-+-{'-'*9}-+-{'-'*11}-+-----")
    results = []
    for k in args.k:
        note = ""
        ok = False
        wall = float("nan")
        r2 = float("nan")
        try:
            tk = time.perf_counter()
            model = gamfit.sae_manifold_fit(
                train,
                K=k,
                d_atom=args.d_atom,
                atom_topology=args.basis,
                assignment=args.assignment,
                n_iter=args.n_iter,
                random_state=args.seed,
                top_k=args.top_k,
                sparsity_weight=args.sparsity_weight,
                smoothness_weight=args.smoothness_weight,
                isometry_weight=args.isometry_weight,
                learning_rate=args.learning_rate,
                ard_per_atom=args.ard,
            )
            wall = time.perf_counter() - tk
            r2 = float(getattr(model, "reconstruction_r2", float("nan")))
            ok = True
            note = model.assignment
        except Exception as e:  # noqa: BLE001
            wall = time.perf_counter() - tk
            note = f"{type(e).__name__}: {e}".replace("\n", " ")[:200]
            traceback.print_exc()
        rss = peak_rss_gb()
        results.append((k, ok, wall, r2, rss, note))
        print(f"{k:>7} | {'YES' if ok else 'NO ':>4} | {wall:>9.2f} | {r2:>9.4f} "
              f"| {rss:>11.3f} | {note}")
        if not ok:
            print(f"# STOP: failed at K={k}; larger K would also fail. See traceback above.")
            break

    print()
    ran = [r for r in results if r[1]]
    if ran:
        max_ran = max(r[0] for r in ran)
        print(f"# largest K that RAN e2e: {max_ran}")
        if max_ran >= 32000:
            print("# ===> MASSIVE-K SAE TARGET REACHED: 32000-atom manifold fit ran e2e.")
        else:
            print(f"# ===> did NOT reach 32000; stalled after K={max_ran}.")
    else:
        print("# ===> NOTHING ran; failed at the smallest K.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
