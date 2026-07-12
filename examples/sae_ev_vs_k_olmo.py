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
  4. For each K on the ladder, fit BOTH a curved-seeded hybrid (circle /
     periodic d=1, with the fitted hybrid-split collapse honored by
     `m.reconstruct`) and a true-linear (`atom_topology="linear"`, d=1)
     dictionary through the production engine, then measure HELD-OUT
     reconstruction EV via `m.reconstruct(X_test)` (frozen decoder re-seated on
     test-row latent coords).
  5. Print the EV(K) table + the hybrid-minus-linear margin per K. The
     discriminating signature (issue's H_flat vs H_curved): curved climbs fast
     then flattens; pure-linear keeps climbing by shattering each curved family
     into ~Theta/(2*sqrt(2*eps)) secants.

CPU PARTIAL: pass --cpu-partial to run K in {1,2,4,8}. It measures whether
curved EV climbs and then starts flattening on the available banked slice, but
it is NOT acceptance for the roadmap.

GPU REQUIREMENT: the official Qwen 32K-reference comparison needs the full
K in {8,32,128,512} ladder at token rate. K>=32 is gated on the device-resident
solver (#1017) + memory work (#1009), so CPU evidence cannot close #1026.

EXAMPLE (OLMo-3-32B base, layer 25, on a cluster compute node):
  python examples/sae_ev_vs_k_olmo.py \
      --npy /path/to/scratch/olmo_data/.../base/activations.npy \
      --olmo-layer 25 --pcs 32 --seed 42

EXAMPLE (generic harvested cache):
  python examples/sae_ev_vs_k_olmo.py --npy qwen3_8b_wikitext/resid_L18.npy \
      --max-rows 2000 --pcs 32 --cpu-partial
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
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
    """PCA fit on TRAIN only; project both; unit-RMS scale from TRAIN only.

    Also returns r, the held-out (TEST) retained-variance ratio: the fraction of
    the centered TEST energy that survives the kept-PC projection. EV measured in
    the PCA-32 coordinate space (EV_Z) overstates a FULL-SPACE reconstruction EV
    by ~1/r, because the PCA arms can never explain the (1 - r) of the test signal
    that the projection threw away. The caller multiplies EV_Z by r to recover a
    full-space EV that is comparable to the official full-space Qwen reference.
    """
    mean = train.mean(axis=0, keepdims=True)
    tc = train - mean
    # economy SVD on the centered train block; right-singular vectors are the PCs.
    _, _, vt = np.linalg.svd(tc, full_matrices=False)
    comp = vt[:pcs].T  # (D, pcs)
    z_tr = tc @ comp
    # UNSCALED, train-mean-centered test projection (before the unit-RMS rescale):
    # r is a pure geometry ratio, so it must be computed on the raw projection.
    tc_te = test - mean
    z_te = tc_te @ comp
    proj_energy = float(np.sum(z_te**2))
    total_energy = float(np.sum(tc_te**2))
    r = proj_energy / total_energy if total_energy > 0.0 else 1.0
    scale = np.sqrt(np.mean(z_tr**2)) or 1.0
    return z_tr / scale, z_te / scale, r


def _ev(target: np.ndarray, fitted: np.ndarray) -> float:
    """Reconstruction explained variance, matching `reconstruction_ev` in the test."""
    resid = target - fitted
    denom = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    if denom <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum(resid**2)) / denom


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sae_fit_worker(
    z_tr,
    z_te,
    k,
    topology,
    seed,
    n_iter,
    separation_barrier_strength,
):
    """TOP-LEVEL (picklable) worker: fit + reconstruct in a child process.

    Lives at module scope so the "spawn" ProcessPoolExecutor can pickle it by
    qualified name (the child re-imports this module). It returns a plain dict of
    EVs/timings/diagnostics — everything already json-able — so nothing exotic has
    to survive the IPC boundary. The PCA arrays z_tr/z_te arrive as numpy arrays.
    """
    from gamfit import sae_manifold_fit

    fit_started = time.perf_counter()
    m = sae_manifold_fit(
        z_tr,
        K=k,
        d_atom=1,
        atom_topology=topology,  # "circle" (curved) or "linear" (true rank-1 affine lane)
        n_iter=n_iter,
        random_state=seed,
        separation_barrier_strength=separation_barrier_strength,
        # This is a fixed-K EV-vs-K sweep on the SHIPPED default fit path: fit
        # exactly `k` atoms and stop. The default bundled pipeline additionally
        # runs a topology race (run_structure_search) and TWO structured-whitening
        # residual passes (structured_residual_passes, default 2) — each re-running
        # the whole outer rho search from scratch — which is what pushed this
        # example past the 900s budget (gh#2267) while adding atoms the sweep does
        # not want. Disable both to take the genuinely unbundled direct path
        # (structured_residual_passes=0 is the load-bearing one — the two passes
        # are NOT reachable via run_structure_search alone).
        run_structure_search=False,
        structured_residual_passes=0,
    )
    fit_seconds = time.perf_counter() - fit_started

    reconstruct_started = time.perf_counter()
    fitted_test = m.reconstruct(z_te)
    reconstruct_seconds = time.perf_counter() - reconstruct_started

    # #1026 IN-SAMPLE probe: reconstruct the SAME train rows (same order ⇒ the
    # hybrid-split collapse-rescue's train-indexed fresh codes align) to isolate
    # whether the curved fit's collapse is repaired in-sample, separately from the
    # held-out generalization (which still needs the OOS coordinate projection).
    train_probe_error = None
    try:
        fitted_train = m.reconstruct(z_tr)
        train_ev = _ev(z_tr, fitted_train)
    except Exception as exc:  # noqa: BLE001 - diagnostic only, never fail the fit
        train_ev = float("nan")
        train_probe_error = str(exc)

    hybrid_split = getattr(m, "hybrid_split", None)
    atom_topologies = [str(v) for v in getattr(m, "atom_topologies", [])]
    return {
        "test_ev": _ev(z_te, fitted_test),
        "train_ev": train_ev,
        "fit_seconds": fit_seconds,
        "reconstruct_seconds": reconstruct_seconds,
        "hybrid_split": None if hybrid_split is None else _jsonable(hybrid_split),
        "atom_topologies": atom_topologies,
        "train_probe_error": train_probe_error,
    }


def _fit_ev(
    z_tr,
    z_te,
    k: int,
    topology: str,
    seed: int,
    n_iter: int,
    separation_barrier_strength: float | None,
) -> tuple[float, float, float, dict | None, list[str]]:
    """Fit one dictionary through the production engine; return HELD-OUT EV.

    The fit + reconstruct run in a "spawn" child process for a clean, isolated
    interpreter, and we block on its result. The fit is bounded by deterministic
    work (`--n-iter` and the solver's own iteration/convergence criteria), not by
    a wall-clock timeout: clipping by elapsed time is non-deterministic and
    machine-dependent (#2055), so a slow fit is fixed or bounded by work rather
    than cut short by a timer.
    """
    ctx = multiprocessing.get_context("spawn")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    try:
        result = executor.submit(
            _sae_fit_worker,
            z_tr,
            z_te,
            k,
            topology,
            seed,
            n_iter,
            separation_barrier_strength,
        ).result()
    finally:
        executor.shutdown(wait=False)

    fit_seconds = result["fit_seconds"]
    reconstruct_seconds = result["reconstruct_seconds"]
    if result["train_probe_error"] is not None:
        print(f"[{topology} K={k}] train-EV probe failed: {result['train_probe_error']}", flush=True)
    print(
        f"[{topology} K={k}] IN_SAMPLE_EV={result['train_ev']:.4f}  "
        f"held_out_EV={result['test_ev']:.4f}",
        flush=True,
    )
    return (
        result["test_ev"],
        fit_seconds,
        reconstruct_seconds,
        result["hybrid_split"],
        result["atom_topologies"],
    )


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
    # All EV comparisons run on the FULL-space values (`*_ev_out` carry r * EV_Z;
    # see FIX 1 / _pca_project): the official reference is a full-space number, so
    # parity must be judged in full space, not in PCA-32 coordinate space.
    measured = {int(row["K"]): row for row in rows}
    required = set(REQUIRED_ACCEPTANCE_LADDER)
    missing = sorted(required - set(ladder))
    full_ladder_measured = required.issubset(measured)

    def _finite(values):
        return [v for v in values if np.isfinite(v)]

    finite_linear = _finite([float(row["linear_ev_out"]) for row in rows])
    finite_hybrid = _finite([float(row["hybrid_ev_out"]) for row in rows])
    best_linear = max(finite_linear) if finite_linear else float("nan")
    best_hybrid = max(finite_hybrid) if finite_hybrid else float("nan")
    # Parity bar = max over the finite comparators (best finite linear arm + the
    # official full-space reference). The official reference is finite, so this can
    # never collapse to NaN the way `max(nan, official)` did before.
    comparators = _finite([best_linear, float(official_reference_ev)])
    parity_ev = max(comparators) if comparators else float("nan")

    # Closure requires a FINITE linear comparator AND a FINITE hybrid value at every
    # required rung — a NaN (failed or timed-out) arm cannot silently pass the gate.
    rungs_missing_finite_linear = [
        k for k in REQUIRED_ACCEPTANCE_LADDER
        if k in measured and not np.isfinite(float(measured[k]["linear_ev_out"]))
    ]
    rungs_missing_finite_hybrid = [
        k for k in REQUIRED_ACCEPTANCE_LADDER
        if k in measured and not np.isfinite(float(measured[k]["hybrid_ev_out"]))
    ]

    closure_allowed = (
        not cpu_partial
        and not missing
        and full_ladder_measured
        and not rungs_missing_finite_linear
        and not rungs_missing_finite_hybrid
        and np.isfinite(best_hybrid)
        and np.isfinite(parity_ev)
        and best_hybrid >= parity_ev
    )
    blocker = None
    if cpu_partial:
        blocker = "CPU partial run; full Qwen/OLMo K={8,32,128,512} ladder was not attempted."
    elif missing:
        blocker = f"Missing required acceptance rung(s): {missing}."
    elif not full_ladder_measured:
        blocker = "Required acceptance ladder was requested but did not finish every rung."
    elif rungs_missing_finite_linear:
        blocker = (
            f"No finite linear held-out EV at required rung(s): {rungs_missing_finite_linear}."
        )
    elif rungs_missing_finite_hybrid:
        blocker = (
            f"No finite curved/hybrid held-out EV at required rung(s): {rungs_missing_finite_hybrid}."
        )
    elif not np.isfinite(best_hybrid):
        blocker = "No finite curved/hybrid held-out EV was measured."
    elif not np.isfinite(parity_ev):
        blocker = "Parity bar is not finite (no finite linear comparator and no finite official reference EV)."
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
        "best_linear_ev": float(best_linear),
        "best_hybrid_ev": float(best_hybrid),
        "parity_ev_bar": float(parity_ev),
        "rungs_missing_finite_linear": [int(k) for k in rungs_missing_finite_linear],
        "rungs_missing_finite_hybrid": [int(k) for k in rungs_missing_finite_hybrid],
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
        "--official-qwen-w32k-ev",
        type=float,
        default=OFFICIAL_QWEN_W32K_EV,
        help="held-out EV of the official Qwen W32K linear-SAE reference from figH",
    )
    ap.add_argument("--out", help="optional JSON file for the measured table")
    ap.add_argument(
        "--sep-mu",
        type=float,
        default=None,
        help="per-fit decoder-repulsion strength (default: evidence-derived)",
    )
    args = ap.parse_args()

    if args.sep_mu is not None:
        print(f"[barrier] sep_mu={args.sep_mu}", flush=True)

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
    z_tr, z_te, r = _pca_project(x[train_idx], x[test_idx], args.pcs)
    ladder = list(CPU_PARTIAL_LADDER) if args.cpu_partial else _parse_ladder(args.k_ladder)

    print(f"=== #1026 real-data EV-vs-K frontier ===")
    print(f"N={n} (train={len(train_idx)}, test={len(test_idx)}), D={x.shape[1]} -> PCA-{args.pcs}, seed={args.seed}")
    print(
        f"PCA held-out retained-variance ratio r={r:.6f}. EV is measured in PCA-{args.pcs} "
        f"coordinate space (EV_Z); the FULL-space EV reported as *_EV_full = r * EV_Z, because "
        f"the kept PCs can never explain the {1.0 - r:.6f} of the centered test signal the "
        f"projection discarded. Closure compares the FULL-space EV against the (full-space) "
        f"official Qwen W32K reference."
    )
    if args.cpu_partial:
        print("CPU partial ladder: K={1,2,4,8}. This run cannot close #1026.")
    else:
        print("Acceptance ladder: K={8,32,128,512}; parity bar includes the official Qwen W32K held-out EV.")
    print(
        f"{'K':>4}  {'hyb_EV_full':>12}  {'hyb_EV_pca':>11}  {'lin_EV_full':>12}  "
        f"{'lin_EV_pca':>11}  {'(h-l)_full':>11}  {'hyb_s':>8}  {'lin_s':>8}  {'rec_s':>8}"
    )
    rows = []
    for k in ladder:
        ev_h_pca, fit_h, recon_h, hybrid_split, hybrid_topologies = _fit_ev(
            z_tr,
            z_te,
            k,
            "circle",
            args.seed,
            args.n_iter,
            args.sep_mu,
        )
        # The pure-linear comparison arm rides a separate basis lane whose OOS /
        # basis_with_jet plumbing can error independently of the curved arm (e.g.
        # the unsupported "linear" basis kind). A failure there must NOT discard
        # the curved EV-vs-K climb — the #1026 headline measurement — so the
        # linear arm is reported as NaN and the row still prints.
        try:
            ev_l_pca, fit_l, recon_l, linear_split, linear_topologies = _fit_ev(
                z_tr,
                z_te,
                k,
                "linear",
                args.seed,
                args.n_iter,
                args.sep_mu,
            )
        except Exception as exc:  # noqa: BLE001 — arm-isolated, reported honestly
            print(f"[linear K={k}] arm failed, reporting NaN: {exc}", flush=True)
            ev_l_pca, fit_l, recon_l = float("nan"), float("nan"), float("nan")
            linear_split, linear_topologies = None, None
        # FULL-space EV = r * EV_Z (see _pca_project): the PCA-space EV is inflated
        # by ~1/r relative to a full-space reconstruction, so the acceptance gate and
        # the official-reference parity must use the full-space numbers.
        ev_h_full = r * ev_h_pca
        ev_l_full = r * ev_l_pca
        print(
            f"{k:>4}  {ev_h_full:>12.6f}  {ev_h_pca:>11.6f}  {ev_l_full:>12.6f}  "
            f"{ev_l_pca:>11.6f}  {ev_h_full - ev_l_full:>11.6f}  "
            f"{fit_h:>8.1f}  {fit_l:>8.1f}  {max(recon_h, recon_l):>8.1f}"
        )
        rows.append(
            {
                "K": k,
                # `*_ev_out` carry the FULL-space values so the acceptance gate uses them.
                "hybrid_ev_out": ev_h_full,
                "linear_ev_out": ev_l_full,
                "hybrid_ev_out_full": ev_h_full,
                "linear_ev_out_full": ev_l_full,
                "hybrid_ev_out_pca": ev_h_pca,
                "linear_ev_out_pca": ev_l_pca,
                "pca_retained_var_ratio": float(r),
                "hybrid_minus_linear": ev_h_full - ev_l_full,
                "hybrid_minus_linear_pca": ev_h_pca - ev_l_pca,
                "hybrid_fit_seconds": fit_h,
                "linear_fit_seconds": fit_l,
                "max_reconstruct_seconds": max(recon_h, recon_l),
                "hybrid_seed_topology": "circle",
                "hybrid_atom_topologies": hybrid_topologies,
                "hybrid_split": hybrid_split,
                "linear_atom_topologies": linear_topologies,
                "linear_split": linear_split,
            }
        )

    print(
        "\nDiscriminating read (issue H_flat vs H_curved): the curved-seeded hybrid should "
        "DOMINATE linear at matched K and CLIMB-then-FLATTEN; pure-linear should keep climbing "
        "by shattering each curved family into ~Theta/(2*sqrt(2*eps)) secants. NOTE: that "
        "~Theta/sqrt(8*eps) count is the MAX-SAGITTA (chord) bound for a fixed worst-case "
        "deviation eps, NOT the MSE/EV law — an EV-loss budget eps gives a milder ~eps^(-1/4) "
        "secant count (and the k=1 rank-1-ray case differs again), so read this as a "
        "qualitative climb-then-flatten vs keep-climbing signature, not an exact EV scaling. "
        "Inspect each row's hybrid_split.atoms for fitted_turning Θ and LOAO ΔEV; the in-tree "
        "predictor for this curve is tests/sae/sae_ev_vs_k_frontier.rs."
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
            "pca_retained_var_ratio": float(r),
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
