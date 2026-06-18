#!/usr/bin/env python
"""#1026 — the real-data EV-vs-K discriminating frontier on OLMo / Qwen activations.

Turnkey driver for the ONE remaining #1026 deliverable that needs a GPU + the
banked real residual-stream activations: the EV-vs-K reconstruction curve on
real LLM activations, curved/hybrid dictionary vs true-linear at matched K. The
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
     true-linear (`atom_topology="linear"`, d=1) dictionary through the production engine, then
     measure HELD-OUT reconstruction EV via `m.reconstruct(X_test)` (frozen
     decoder re-seated on test-row latent coords).
  5. Print the EV(K) table + the curved-minus-linear margin per K. The
     discriminating signature (issue's H_flat vs H_curved): curved climbs fast
     then flattens; pure-linear keeps climbing by shattering each curved family
     into ~Theta/(2*sqrt(2*eps)) secants.

CPU PARTIAL: pass --cpu-partial to run K in {1,2,4,8}. It measures whether
curved EV climbs and then starts flattening on the available banked slice, but
it is NOT acceptance for the roadmap.

GPU REQUIREMENT: the official Qwen 32K-reference comparison needs the full
K in {8,32,128,512} ladder at token rate. K>=32 is gated on the device-resident
solver (#1017) + memory work (#1009), so CPU evidence cannot close #1026.

EXAMPLE (OLMo-3-32B base, layer 25, on an MSI compute node):
  python examples/sae_ev_vs_k_olmo.py \
      --npy /path/to/scratch/olmo_data/.../base/activations.npy \
      --olmo-layer 25 --pcs 32 --seed 42

EXAMPLE (generic harvested cache):
  python examples/sae_ev_vs_k_olmo.py --npy qwen3_8b_wikitext/resid_L18.npy \
      --max-rows 2000 --pcs 32 --cpu-partial
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

REQUIRED_ACCEPTANCE_LADDER = (8, 32, 128, 512)
CPU_PARTIAL_LADDER = (1, 2, 4, 8)
OFFICIAL_QWEN_W32K_EV = 0.523


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
        atom_topology=topology,  # "circle" (curved) or "linear" (true rank-1 affine lane)
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


def _parse_ladder(value: str) -> list[int]:
    ladder = [int(s) for s in value.split(",") if s.strip()]
    if not ladder:
        raise SystemExit("--k-ladder must contain at least one K")
    if any(k <= 0 for k in ladder):
        raise SystemExit("--k-ladder values must be positive")
    if len(set(ladder)) != len(ladder):
        raise SystemExit("--k-ladder must not contain duplicate K values")
    return ladder


def _acceptance_report(
    rows: list[dict],
    *,
    ladder: list[int],
    cpu_partial: bool,
    official_reference_ev: float,
) -> dict:
    measured = {int(row["K"]): row for row in rows}
    required = set(REQUIRED_ACCEPTANCE_LADDER)
    missing = sorted(required - set(ladder))
    full_ladder_measured = required.issubset(measured)
    best_curved = max((float(row["curved_ev_out"]) for row in rows), default=float("nan"))
    best_linear = max((float(row["linear_ev_out"]) for row in rows), default=float("nan"))
    best_hybrid = max((float(row["hybrid_ev_out"]) for row in rows), default=float("nan"))
    parity_ev = max(best_linear, official_reference_ev)
    closure_allowed = (
        not cpu_partial
        and not missing
        and full_ladder_measured
        and np.isfinite(best_hybrid)
        and best_hybrid >= parity_ev
    )
    blocker = None
    if cpu_partial:
        blocker = "CPU partial run; full Qwen/OLMo K={8,32,128,512} ladder was not attempted."
    elif missing:
        blocker = f"Missing required acceptance rung(s): {missing}."
    elif not full_ladder_measured:
        blocker = "Required acceptance ladder was requested but did not finish every rung."
    elif not np.isfinite(best_hybrid):
        blocker = "No finite curved/hybrid held-out EV was measured."
    elif best_hybrid < parity_ev:
        blocker = (
            f"Best curved/hybrid EV {best_hybrid:.6f} is below parity bar "
            f"{parity_ev:.6f}."
        )
    return {
        "required_ladder": list(REQUIRED_ACCEPTANCE_LADDER),
        "requested_ladder": [int(k) for k in ladder],
        "official_qwen_w32k_ev": float(official_reference_ev),
        "full_ladder_measured": bool(full_ladder_measured),
        "best_curved_ev": float(best_curved),
        "best_linear_ev": float(best_linear),
        "best_hybrid_ev": float(best_hybrid),
        "parity_ev_bar": float(parity_ev),
        "closure_allowed": bool(closure_allowed),
        "blocker": blocker,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_argument_group("activation source")
    src.add_argument("--npy", help="OLMo activations.npy (3D -> needs --olmo-layer) or 2D (N,D)")
    src.add_argument("--olmo-layer", type=int, default=None, help="layer index for 3D OLMo npy (25 self/qualia, 44 color)")
    src.add_argument("--pt", help="generic harvested .pt cache with key 'X' (N, d_model)")
    ap.add_argument("--pcs", type=int, default=32, help="PCA components kept (figH top-PC budget)")
    ap.add_argument("--k-ladder", default="8,32,128,512", help="comma list of K; #1026 acceptance requires 8,32,128,512")
    ap.add_argument(
        "--cpu-partial",
        action="store_true",
        help="run the documented CPU diagnostic ladder K=1,2,4,8 and mark output as non-closeable",
    )
    ap.add_argument("--max-rows", type=int, default=None, help="deterministic CPU diagnostic subsample before the train/test split")
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
    ap.add_argument(
        "--official-qwen-w32k-ev",
        type=float,
        default=OFFICIAL_QWEN_W32K_EV,
        help="held-out EV of the official Qwen W32K linear-SAE reference from figH",
    )
    ap.add_argument("--out", help="optional JSON file for the measured table")
    args = ap.parse_args()

    x = _load_activations(args)
    rng = np.random.default_rng(args.seed)
    if args.max_rows is not None and args.max_rows < x.shape[0]:
        if args.max_rows < 8:
            raise SystemExit("--max-rows must be at least 8")
        keep = rng.choice(x.shape[0], size=args.max_rows, replace=False)
        keep.sort()
        x = np.asarray(x[keep], dtype=np.float64)
    n = x.shape[0]
    perm = rng.permutation(n)
    n_test = max(1, int(round(args.test_frac * n)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    z_tr, z_te = _pca_project(x[train_idx], x[test_idx], args.pcs)
    ladder = list(CPU_PARTIAL_LADDER) if args.cpu_partial else _parse_ladder(args.k_ladder)

    print(f"=== #1026 real-data EV-vs-K frontier ===")
    print(f"N={n} (train={len(train_idx)}, test={len(test_idx)}), D={x.shape[1]} -> PCA-{args.pcs}, seed={args.seed}")
    if args.cpu_partial:
        print("CPU partial ladder: K={1,2,4,8}. This run cannot close #1026.")
    else:
        print("Acceptance ladder: K={8,32,128,512}; parity bar includes the official Qwen W32K held-out EV.")
    print(
        f"{'K':>4}  {'curved_EV_out':>13}  {'linear_EV_out':>13}  "
        f"{'(curved - linear)':>17}  {'curved_s':>9}  {'linear_s':>9}  {'recon_s':>9}"
    )
    rows = []
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
            "linear",
            args.seed,
            args.n_iter,
            k_max_fit_seconds,
            args.max_reconstruct_seconds,
        )
        print(
            f"{k:>4}  {ev_c:>13.6f}  {ev_l:>13.6f}  {ev_c - ev_l:>17.6f}  "
            f"{fit_c:>9.1f}  {fit_l:>9.1f}  {max(recon_c, recon_l):>9.1f}"
        )
        rows.append(
            {
                "K": k,
                "curved_ev_out": ev_c,
                "hybrid_ev_out": ev_c,
                "linear_ev_out": ev_l,
                "curved_minus_linear": ev_c - ev_l,
                "curved_fit_seconds": fit_c,
                "linear_fit_seconds": fit_l,
                "max_reconstruct_seconds": max(recon_c, recon_l),
            }
        )

    print(
        "\nDiscriminating read (issue H_flat vs H_curved): curved should DOMINATE linear at "
        "matched K and CLIMB-then-FLATTEN; pure-linear should keep climbing by shattering each "
        "curved family into ~Theta/(2*sqrt(2*eps)) secants. The in-tree predictor for this curve "
        "is tests/sae/sae_ev_vs_k_frontier.rs."
    )
    acceptance = _acceptance_report(
        rows,
        ladder=ladder,
        cpu_partial=bool(args.cpu_partial),
        official_reference_ev=args.official_qwen_w32k_ev,
    )
    print(
        "\n#1026 acceptance: "
        f"best_hybrid_EV={acceptance['best_hybrid_ev']:.6f}, "
        f"parity_bar={acceptance['parity_ev_bar']:.6f}, "
        f"closure_allowed={acceptance['closure_allowed']}"
    )
    if acceptance["blocker"] is not None:
        print(f"Closure blocker: {acceptance['blocker']}")
    if args.out:
        payload = {
            "issue": 1026,
            "source": args.npy or args.pt,
            "olmo_layer": args.olmo_layer,
            "n": int(n),
            "train_n": int(len(train_idx)),
            "test_n": int(len(test_idx)),
            "input_dim": int(x.shape[1]),
            "pcs": int(args.pcs),
            "max_rows": args.max_rows,
            "seed": int(args.seed),
            "n_iter": int(args.n_iter),
            "cpu_partial": bool(args.cpu_partial),
            "acceptance": acceptance,
            "rows": rows,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"JSON -> {args.out}")


if __name__ == "__main__":
    main()
