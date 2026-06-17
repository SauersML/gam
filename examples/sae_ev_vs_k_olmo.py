#!/usr/bin/env python
"""#1026 — the real-data EV-vs-K discriminating frontier on OLMo / Qwen activations.

Turnkey driver for the ONE remaining #1026 deliverable that needs a GPU + the
banked real residual-stream activations: the EV-vs-K reconstruction curve on
real LLM activations, curved/hybrid dictionary vs pure-linear at matched K. The
in-tree mechanism (collapsed-linear lane + hybrid split, EV-vs-K frontier test,
distilled amortized encoder) is already LANDED and tested under `tests/sae/`;
this script is the runbook that produces the external measurement those tests
predict, on real data, the moment compute is available.

It is a thin numeric adapter (the #977 boundary: activations are just a response
matrix). NO fitting math lives here — it calls `gamfit.sae_manifold_fit` and
`ManifoldSAE.reconstruct`, the same production entry the frontier test drives.

PROTOCOL (matches the real-data numbers posted to #1026):
  1. Load the activation slice. For OLMo-3-32B the input is
     `activations.npy[:, LAYER, :]` (635 x 5120); LAYER=25 for self/qualia,
     LAYER=44 for color. For a generic harvest (`harvest_residual_activations.py`)
     pass the `(n_tokens, d_model)` cache via --npy / --pt.
  2. 80/20 train/test split (seeded). NO leakage.
  3. PCA fit on TRAIN ONLY, keep top --pcs components (default 32 = figH top-PC
     budget). Project both splits; global scale from TRAIN only (unit RMS).
  4. For each K on the ladder, fit BOTH a curved (circle / periodic d=1) and a
     pure-linear (euclidean d=1) dictionary through the production engine, then
     measure HELD-OUT reconstruction EV via `m.reconstruct(X_test)` (frozen
     decoder re-seated on test-row latent coords).
  5. Print the EV(K) table + the curved-minus-linear margin per K. The
     discriminating signature (issue's H_flat vs H_curved): curved climbs fast
     then flattens; pure-linear keeps climbing by shattering each curved family
     into ~Theta/(2*sqrt(2*eps)) secants.

GPU REQUIREMENT: K in {1,2} runs on CPU in minutes (the banked numbers:
K=1 ~67s, K=2 ~350-410s on an MSI compute node). K>=4 / large-K and the full
{8,32,128,512} ladder need the device-resident solver (#1017) + memory (#1009)
to run at token rate. Until then this driver runs the K in {1,2} arm everywhere
and fails loudly if a requested rung is not production-ready.

EXAMPLE (OLMo-3-32B base, layer 25, on an MSI compute node):
  python examples/sae_ev_vs_k_olmo.py \
      --npy /path/to/scratch/olmo_data/.../base/activations.npy \
      --olmo-layer 25 --pcs 32 --k-ladder 1,2 --seed 42

EXAMPLE (generic harvested cache):
  python examples/sae_ev_vs_k_olmo.py --pt qwen05_wikitext_l12.pt --pcs 32 --k-ladder 1,2
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def _load_activations(args: argparse.Namespace) -> np.ndarray:
    """Return the (N, D) activation matrix to feed the SAE."""
    if args.npy is not None:
        arr = np.load(args.npy)
        if arr.ndim == 3:
            # OLMo `activations.npy` is (prompts, layers, d_model).
            if args.olmo_layer is None:
                raise SystemExit(
                    "3D activations.npy needs --olmo-layer (e.g. 25 for self/qualia, 44 for color)"
                )
            arr = arr[:, args.olmo_layer, :]
        return np.asarray(arr, dtype=np.float64)
    if args.pt is not None:
        import torch

        blob = torch.load(args.pt, map_location="cpu")
        x = blob["X"] if isinstance(blob, dict) else blob
        return np.asarray(x, dtype=np.float64)
    raise SystemExit("provide --npy (OLMo activations.npy) or --pt (harvested cache)")


def _pca_project(train: np.ndarray, test: np.ndarray, pcs: int):
    """PCA fit on TRAIN only; project both; unit-RMS scale from TRAIN only."""
    mean = train.mean(axis=0, keepdims=True)
    tc = train - mean
    # economy SVD on the centered train block; right-singular vectors are the PCs.
    _, _, vt = np.linalg.svd(tc, full_matrices=False)
    comp = vt[:pcs].T  # (D, pcs)
    z_tr = tc @ comp
    z_te = (test - mean) @ comp
    scale = np.sqrt(np.mean(z_tr**2)) or 1.0
    return z_tr / scale, z_te / scale


def _ev(target: np.ndarray, fitted: np.ndarray) -> float:
    """Reconstruction explained variance, matching `reconstruction_ev` in the test."""
    resid = target - fitted
    denom = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    if denom <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum(resid**2)) / denom


def _fit_ev(
    z_tr,
    z_te,
    k: int,
    topology: str,
    seed: int,
    n_iter: int,
    max_fit_seconds: float,
    max_reconstruct_seconds: float,
) -> tuple[float, float, float]:
    """Fit one dictionary through the production engine; return HELD-OUT EV."""
    from gamfit import sae_manifold_fit

    fit_started = time.perf_counter()
    m = sae_manifold_fit(
        z_tr,
        K=k,
        d_atom=1,
        atom_topology=topology,  # "circle" (curved) or "euclidean" (degree-2 quadratic patch, #1201 — NOT a linear atom)
        assignment="ibp_map",
        n_iter=n_iter,
        random_state=seed,
    )
    fit_seconds = time.perf_counter() - fit_started
    if fit_seconds > max_fit_seconds:
        raise SystemExit(
            f"{topology} K={k} fit exceeded wall-clock guard: "
            f"{fit_seconds:.1f}s > {max_fit_seconds:.1f}s"
        )

    reconstruct_started = time.perf_counter()
    fitted_test = m.reconstruct(z_te)
    reconstruct_seconds = time.perf_counter() - reconstruct_started
    if reconstruct_seconds > max_reconstruct_seconds:
        raise SystemExit(
            f"{topology} K={k} held-out reconstruct exceeded wall-clock guard: "
            f"{reconstruct_seconds:.1f}s > {max_reconstruct_seconds:.1f}s"
        )
    return _ev(z_te, fitted_test), fit_seconds, reconstruct_seconds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_argument_group("activation source")
    src.add_argument("--npy", help="OLMo activations.npy (3D -> needs --olmo-layer) or 2D (N,D)")
    src.add_argument("--olmo-layer", type=int, default=None, help="layer index for 3D OLMo npy (25 self/qualia, 44 color)")
    src.add_argument("--pt", help="generic harvested .pt cache with key 'X' (N, d_model)")
    ap.add_argument("--pcs", type=int, default=32, help="PCA components kept (figH top-PC budget)")
    ap.add_argument("--k-ladder", default="1,2", help="comma list of K (e.g. 1,2 or 8,32,128,512)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument(
        "--max-fit-seconds",
        type=float,
        default=900.0,
        help=(
            "PER-ATOM wall-clock guard, scaled by K (the K>=2 joint inner solve + "
            "the inter-atom routing-collapse-protected outer homotopy walk both grow "
            "with K, so a fixed budget that fits K=1 (~67s) wrongly trips K=2 (~1245s). "
            "The effective guard for a rung is max-fit-seconds * K."
        ),
    )
    ap.add_argument("--max-reconstruct-seconds", type=float, default=60.0)
    args = ap.parse_args()

    x = _load_activations(args)
    n = x.shape[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(args.test_frac * n)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    z_tr, z_te = _pca_project(x[train_idx], x[test_idx], args.pcs)
    ladder = [int(s) for s in args.k_ladder.split(",") if s.strip()]

    print(f"=== #1026 real-data EV-vs-K frontier ===")
    print(f"N={n} (train={len(train_idx)}, test={len(test_idx)}), D={x.shape[1]} -> PCA-{args.pcs}, seed={args.seed}")
    print(
        f"{'K':>4}  {'curved_EV_out':>13}  {'linear_EV_out':>13}  "
        f"{'(curved - linear)':>17}  {'curved_s':>9}  {'linear_s':>9}  {'recon_s':>9}"
    )
    for k in ladder:
        # Per-K wall-clock guard: the K>=2 fit's joint inner solve and the
        # routing-collapse-protected outer homotopy walk both grow with K, so the
        # budget must scale with K rather than gate every rung on the K=1 time.
        k_max_fit_seconds = args.max_fit_seconds * k
        ev_c, fit_c, recon_c = _fit_ev(
            z_tr,
            z_te,
            k,
            "circle",
            args.seed,
            args.n_iter,
            k_max_fit_seconds,
            args.max_reconstruct_seconds,
        )
        ev_l, fit_l, recon_l = _fit_ev(
            z_tr,
            z_te,
            k,
            "euclidean",
            args.seed,
            args.n_iter,
            k_max_fit_seconds,
            args.max_reconstruct_seconds,
        )
        print(
            f"{k:>4}  {ev_c:>13.6f}  {ev_l:>13.6f}  {ev_c - ev_l:>17.6f}  "
            f"{fit_c:>9.1f}  {fit_l:>9.1f}  {max(recon_c, recon_l):>9.1f}"
        )

    print(
        "\nDiscriminating read (issue H_flat vs H_curved): curved should DOMINATE linear at "
        "matched K and CLIMB-then-FLATTEN; pure-linear should keep climbing by shattering each "
        "curved family into ~Theta/(2*sqrt(2*eps)) secants. The in-tree predictor for this curve "
        "is tests/sae/sae_ev_vs_k_frontier.rs."
    )


if __name__ == "__main__":
    main()
