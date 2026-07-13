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
matrix). NO fitting math lives here. The benchmark routes each arm through the
production engine that owns its actual model contract:

  * K <= P: both arms use the dense certified ManifoldSAE lane (softmax gates),
    differing only in curved-circle versus true-linear atom topology.
  * K > P curved: ManifoldSAE's explicitly admitted hard-TopK curved lane.
  * K > P linear: SparseDictionaryFit's fixed-width sparse linear lane.

The overcomplete arms share exactly K and the per-row active support s, but not
model type, precision, optimizer, reconstruction API, or certificate family.

PROTOCOL (matches the real-data numbers posted to #1026):
  1. Load the activation slice. For OLMo-3-32B the input is
     `activations.npy[:, LAYER, :]` (635 x 5120); LAYER=25 for self/qualia,
     LAYER=44 for color. For a generic harvest (`harvest_residual_activations.py`)
     pass the `(n_tokens, d_model)` cache via --npy / --pt.
  2. 80/20 train/test split (seeded). NO leakage.
  3. PCA fit on TRAIN ONLY, keep top --pcs components (default 32 = figH top-PC
     budget). Project both splits; global scale from TRAIN only (unit RMS).
  4. For each K on the ladder, fit BOTH a curved-seeded model and a true-linear
     model. At K <= P they are dense certified ManifoldSAEs. At K > P, curved
     uses hard TopK with exactly --active-support assignments per row, while
     linear uses SparseDictionaryFit with exactly the same selected support
     width. Each arm performs its own production frozen-model OOS encode before
     reconstruction; the two certificate families remain explicitly distinct.
  5. Print the EV(K) table + the hybrid-minus-linear margin per K. The
     discriminating signature (issue's H_flat vs H_curved): curved climbs fast
     then flattens; pure-linear keeps climbing by shattering each curved family
     into ~Theta/(2*sqrt(2*eps)) secants.

CPU PARTIAL: pass --cpu-partial to run K in {1,2,4,8}. It measures whether
curved EV climbs and then starts flattening on the available banked slice, but
it is NOT acceptance for the roadmap. The full K in {8,32,128,512} ladder is
also a supported CPU production path and is the authoritative timing contract
for the committed 635-row fixture (#2267). A GPU can accelerate larger token-
rate corpora, but it is not a validity requirement for this measurement.

The authoritative #2267 CPU timing fixes --linear-score-mode=off (the default)
so no accelerator fallback or machine-dependent route choice enters the timing.
Use --linear-score-mode=required only for a deliberately GPU-resident sparse-
linear measurement; it fails closed if the device route is not admitted.

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
DEFAULT_ACTIVE_SUPPORT = 1

DENSE_MANIFOLD_LANE = "dense_manifold_softmax"
CURVED_TOPK_LANE = "curved_manifold_topk"
LINEAR_SPARSE_LANE = "linear_sparse_dictionary"


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


def _arm_route(k: int, p: int, arm: str, active_support: int) -> dict:
    """Return the one permitted production route for a benchmark arm.

    Dense manifold certification owns K <= P. Overcomplete fits are different
    statistical objects: a curved hard-TopK ManifoldSAE and a sparse linear
    dictionary. Keeping this decision pure makes accidental dense K > P calls
    impossible to hide in worker/process plumbing.
    """
    for value, label in ((k, "K"), (p, "P"), (active_support, "active support")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{label} must be an integer; got {value!r}")
        if int(value) <= 0:
            raise ValueError(f"{label} must be positive; got {value!r}")
    if arm not in {"curved", "linear"}:
        raise ValueError(f"arm must be 'curved' or 'linear'; got {arm!r}")
    if active_support > k:
        raise ValueError(
            f"active support s={active_support} cannot exceed dictionary width K={k}"
        )

    if k <= p:
        return {
            "arm": arm,
            "lane": DENSE_MANIFOLD_LANE,
            "model_type": "ManifoldSAE",
            "certificate_type": "ManifoldSAE.certificates+termination",
            "assignment": "softmax",
            "active_support": None,
            "support_contract": "dense_full_support",
            "precision": "float64",
        }
    if arm == "curved":
        return {
            "arm": arm,
            "lane": CURVED_TOPK_LANE,
            "model_type": "ManifoldSAE",
            "certificate_type": "ManifoldSAE.certificates+termination",
            "assignment": "topk",
            "active_support": int(active_support),
            "support_contract": "exact_selected_support_values_per_row",
            "precision": "float64",
        }
    return {
        "arm": arm,
        "lane": LINEAR_SPARSE_LANE,
        "model_type": "SparseDictionaryFit",
        "certificate_type": "SparseDictionaryConvergence",
        "assignment": "active_set_least_squares",
        "active_support": int(active_support),
        "support_contract": "exact_selected_nonzero_codes_per_row",
        "precision": "float32",
    }


def _exact_dense_support_summary(
    assignments,
    *,
    rows: int,
    k: int,
    active_support: int,
    label: str,
) -> dict:
    values = np.asarray(assignments, dtype=np.float64)
    if values.shape != (rows, k):
        raise RuntimeError(
            f"{label} assignments have shape {values.shape}; expected {(rows, k)}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label} assignments contain non-finite values")
    live = np.count_nonzero(values, axis=1)
    if np.any(live != active_support):
        raise RuntimeError(
            f"{label} violated exact TopK support s={active_support}; "
            f"observed per-row nonzeros in [{int(live.min())}, {int(live.max())}]"
        )
    return {
        "rows": int(rows),
        "selected_min": int(live.min()),
        "selected_max": int(live.max()),
        "nonzero_min": int(live.min()),
        "nonzero_max": int(live.max()),
    }


def _exact_sparse_support_summary(
    indices,
    codes,
    *,
    rows: int,
    k: int,
    active_support: int,
    label: str,
) -> dict:
    idx = np.asarray(indices)
    values = np.asarray(codes, dtype=np.float32)
    expected = (rows, active_support)
    if idx.shape != expected or values.shape != expected:
        raise RuntimeError(
            f"{label} sparse route has indices {idx.shape} and codes {values.shape}; "
            f"expected {expected} for both"
        )
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label} sparse codes contain non-finite values")
    if np.any(idx < 0) or np.any(idx >= k):
        raise RuntimeError(f"{label} sparse route contains an out-of-range atom index")
    selected = np.asarray([len(set(row.tolist())) for row in idx], dtype=np.int64)
    nonzero = np.count_nonzero(values, axis=1)
    if np.any(selected != active_support) or np.any(nonzero != active_support):
        raise RuntimeError(
            f"{label} violated exact sparse support s={active_support}; "
            f"distinct selected atoms [{int(selected.min())}, {int(selected.max())}], "
            f"nonzero codes [{int(nonzero.min())}, {int(nonzero.max())}]"
        )
    return {
        "rows": int(rows),
        "selected_min": int(selected.min()),
        "selected_max": int(selected.max()),
        "nonzero_min": int(nonzero.min()),
        "nonzero_max": int(nonzero.max()),
    }


def _manifold_fit_worker(
    z_tr,
    z_te,
    k,
    arm,
    assignment,
    active_support,
    seed,
    manifold_iterations,
    separation_barrier_strength,
):
    """Fit one dense/curved manifold arm and run coherent OOS inference."""
    from gamfit import sae_manifold_fit

    topology = "circle" if arm == "curved" else "linear"
    fit_options = {}
    if assignment == "topk":
        fit_options = {"assignment": "topk", "top_k": int(active_support)}
    elif assignment != "softmax":
        raise RuntimeError(f"unsupported manifold assignment route {assignment!r}")

    fit_started = time.perf_counter()
    if assignment != "topk":
        fit_options["separation_barrier_strength"] = separation_barrier_strength
    m = sae_manifold_fit(
        z_tr,
        K=k,
        d_atom=1,
        atom_topology=topology,  # "circle" (curved) or "linear" (true rank-1 affine lane)
        n_iter=manifold_iterations,
        random_state=seed,
        # This is a fixed-K EV-vs-K sweep: fit exactly `k` atoms and stop. Keep
        # both optional pipeline stages explicit so this example cannot silently
        # turn back into topology search plus repeated full outer refits (#2267).
        run_structure_search=False,
        structured_residual_passes=0,
        **fit_options,
    )
    fit_seconds = time.perf_counter() - fit_started

    fitted_width = getattr(m, "requested_k", getattr(m, "chosen_k", None))
    if fitted_width != k:
        raise RuntimeError(
            f"{arm} manifold route returned {type(m).__name__} with dictionary width={fitted_width!r}; "
            f"fixed-K benchmark requires a ManifoldSAE requested at K={k}"
        )

    reconstruct_started = time.perf_counter()
    train_latents = dict(m.converged_latents(z_tr))
    test_latents = dict(m.converged_latents(z_te))
    reconstruct_seconds = time.perf_counter() - reconstruct_started
    fitted_train = np.asarray(train_latents["fitted"], dtype=np.float64)
    fitted_test = np.asarray(test_latents["fitted"], dtype=np.float64)
    if fitted_train.shape != z_tr.shape or fitted_test.shape != z_te.shape:
        raise RuntimeError(
            f"{arm} manifold reconstruction shape mismatch: train {fitted_train.shape} "
            f"vs {z_tr.shape}, held-out {fitted_test.shape} vs {z_te.shape}"
        )
    train_ev = _ev(z_tr, fitted_train)
    test_ev = _ev(z_te, fitted_test)
    if not np.isfinite(train_ev) or not np.isfinite(test_ev):
        raise RuntimeError(
            f"{arm} K={k} produced non-finite reconstruction EV "
            f"(train={train_ev!r}, held_out={test_ev!r})"
        )

    support = None
    if assignment == "topk":
        support = {
            "train": _exact_sparse_support_summary(
                train_latents["support_indices"],
                train_latents["support_values"],
                rows=z_tr.shape[0],
                k=k,
                active_support=active_support,
                label=f"{arm} train",
            ),
            "held_out": _exact_sparse_support_summary(
                test_latents["support_indices"],
                test_latents["support_values"],
                rows=z_te.shape[0],
                k=k,
                active_support=active_support,
                label=f"{arm} held-out",
            ),
        }

    hybrid_split = getattr(m, "hybrid_split", None)
    atom_topologies = [str(v) for v in getattr(m, "atom_topologies", [])]
    return {
        "test_ev": test_ev,
        "train_ev": train_ev,
        "fit_seconds": fit_seconds,
        "reconstruct_seconds": reconstruct_seconds,
        "hybrid_split": None if hybrid_split is None else _jsonable(hybrid_split),
        "atom_topologies": atom_topologies,
        "support_evidence": support,
        "certificate": {
            "type": "ManifoldSAE.certificates+termination",
            "certificates": _jsonable(getattr(m, "certificates", None)),
            "termination": _jsonable(getattr(m, "termination", None)),
        },
    }


def _sparse_convergence_payload(convergence) -> dict:
    names = (
        "inner_ev_residual",
        "inner_tolerance",
        "decoder_residual",
        "decoder_tolerance",
        "routing_residual",
        "routing_tolerance",
        "outer_rho_residual",
        "outer_tolerance",
        "selected_rho",
        "outer_iterations",
    )
    return {name: _jsonable(getattr(convergence, name)) for name in names}


def _linear_sparse_fit_worker(
    z_tr,
    z_te,
    k,
    active_support,
    linear_epochs,
    score_mode,
):
    """Fit the production sparse-linear lane and explicitly route both splits."""
    from gamfit import sparse_dictionary_fit

    fit_started = time.perf_counter()
    m = sparse_dictionary_fit(
        z_tr,
        K=k,
        active=active_support,
        max_epochs=linear_epochs,
        score_mode=score_mode,
    )
    fit_seconds = time.perf_counter() - fit_started

    decoder = np.asarray(getattr(m, "decoder", None))
    if type(m).__name__ != "SparseDictionaryFit" or decoder.shape != (k, z_tr.shape[1]):
        raise RuntimeError(
            f"linear sparse route returned {type(m).__name__} with decoder "
            f"shape {decoder.shape}; expected SparseDictionaryFit {(k, z_tr.shape[1])}"
        )
    if getattr(m, "active", None) != active_support:
        raise RuntimeError(
            f"linear sparse fit reported active={getattr(m, 'active', None)!r}; "
            f"expected exact support s={active_support}"
        )

    reconstruct_started = time.perf_counter()
    train_route = m.transform(z_tr, active=active_support, score_mode=score_mode)
    test_route = m.transform(z_te, active=active_support, score_mode=score_mode)
    train_support = _exact_sparse_support_summary(
        train_route.indices,
        train_route.codes,
        rows=z_tr.shape[0],
        k=k,
        active_support=active_support,
        label="linear train",
    )
    test_support = _exact_sparse_support_summary(
        test_route.indices,
        test_route.codes,
        rows=z_te.shape[0],
        k=k,
        active_support=active_support,
        label="linear held-out",
    )
    fitted_train = np.asarray(
        m.reconstruct(train_route.indices, train_route.codes), dtype=np.float64
    )
    fitted_test = np.asarray(
        m.reconstruct(test_route.indices, test_route.codes), dtype=np.float64
    )
    reconstruct_seconds = time.perf_counter() - reconstruct_started
    if fitted_train.shape != z_tr.shape or fitted_test.shape != z_te.shape:
        raise RuntimeError(
            f"linear sparse reconstruction shape mismatch: train {fitted_train.shape} "
            f"vs {z_tr.shape}, held-out {fitted_test.shape} vs {z_te.shape}"
        )
    train_ev = _ev(z_tr, fitted_train)
    test_ev = _ev(z_te, fitted_test)
    if not np.isfinite(train_ev) or not np.isfinite(test_ev):
        raise RuntimeError(
            f"linear sparse K={k} produced non-finite reconstruction EV "
            f"(train={train_ev!r}, held_out={test_ev!r})"
        )

    return {
        "test_ev": test_ev,
        "train_ev": train_ev,
        "fit_seconds": fit_seconds,
        "reconstruct_seconds": reconstruct_seconds,
        "hybrid_split": None,
        "atom_topologies": ["linear"] * k,
        "support_evidence": {"train": train_support, "held_out": test_support},
        "certificate": {
            "type": "SparseDictionaryConvergence",
            "convergence": _sparse_convergence_payload(m.convergence),
        },
        "score_route_stats": {
            "fit": _jsonable(m.score_route_stats),
            "train_transform": _jsonable(train_route.score_route_stats),
            "held_out_transform": _jsonable(test_route.score_route_stats),
        },
    }


def _fit_ev(
    z_tr,
    z_te,
    k: int,
    arm: str,
    seed: int,
    manifold_iterations: int,
    linear_epochs: int,
    active_support: int,
    linear_score_mode: str,
    separation_barrier_strength: float | None,
) -> dict:
    """Fit one dictionary through its admitted production engine.

    The fit + reconstruct run in a "spawn" child process for a clean, isolated
    interpreter, and we block on its result. The fit is bounded by deterministic
    work (manifold iterations / sparse epochs plus solver convergence criteria),
    not by a wall-clock timeout. Clipping by elapsed time is non-deterministic
    and machine-dependent (#2055).
    """
    route = _arm_route(k, z_tr.shape[1], arm, active_support)
    ctx = multiprocessing.get_context("spawn")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    try:
        if route["lane"] == LINEAR_SPARSE_LANE:
            future = executor.submit(
                _linear_sparse_fit_worker,
                z_tr,
                z_te,
                k,
                active_support,
                linear_epochs,
                linear_score_mode,
            )
        else:
            future = executor.submit(
                _manifold_fit_worker,
                z_tr,
                z_te,
                k,
                arm,
                route["assignment"],
                active_support,
                seed,
                manifold_iterations,
                separation_barrier_strength,
            )
        result = future.result()
    finally:
        executor.shutdown(wait=False)

    result["route"] = route
    print(
        f"[{arm} K={k} lane={route['lane']}] IN_SAMPLE_EV={result['train_ev']:.4f}  "
        f"held_out_EV={result['test_ev']:.4f}",
        flush=True,
    )
    return result


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
    ap.add_argument(
        "--active-support",
        type=int,
        default=DEFAULT_ACTIVE_SUPPORT,
        help=(
            "exact per-row support s shared by both K>P arms; must be no larger "
            "than the smallest requested K"
        ),
    )
    ap.add_argument(
        "--manifold-iterations",
        type=int,
        default=40,
        help="iteration budget for ManifoldSAE arms",
    )
    ap.add_argument(
        "--linear-epochs",
        type=int,
        default=30,
        help="epoch budget for overcomplete SparseDictionaryFit linear arms",
    )
    ap.add_argument(
        "--linear-score-mode",
        choices=("off", "required"),
        default="off",
        help=(
            "sparse-linear scorer route: 'off' is authoritative reproducible CPU "
            "timing; 'required' fails closed unless GPU scoring is admitted"
        ),
    )
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

    if args.pcs <= 0:
        raise SystemExit("--pcs must be positive")
    if args.manifold_iterations <= 0:
        raise SystemExit("--manifold-iterations must be positive")
    if args.linear_epochs <= 0:
        raise SystemExit("--linear-epochs must be positive")
    if not 0.0 < args.test_frac < 1.0:
        raise SystemExit("--test-frac must lie strictly between 0 and 1")

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
    if args.active_support <= 0:
        raise SystemExit("--active-support must be positive")
    if args.active_support > min(ladder):
        raise SystemExit(
            f"--active-support={args.active_support} exceeds the smallest requested "
            f"dictionary width K={min(ladder)}"
        )
    realized_p = int(z_tr.shape[1])

    print(f"=== #1026 real-data EV-vs-K frontier ===")
    print(
        f"N={n} (train={len(train_idx)}, test={len(test_idx)}), D={x.shape[1]} "
        f"-> requested PCA-{args.pcs}, realized P={realized_p}, seed={args.seed}"
    )
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
        "Route contract: K<=P uses dense ManifoldSAE/softmax for both arms. "
        f"K>P uses curved ManifoldSAE/TopK(s={args.active_support}) versus "
        f"linear SparseDictionaryFit(s={args.active_support}, "
        f"score_mode={args.linear_score_mode!r}). K, data split, held-out EV metric, "
        "and overcomplete selected support are matched; model type, precision, "
        "optimizer, and certificate are intentionally lane-specific."
    )
    print(
        f"{'K':>4}  {'hyb_EV_full':>12}  {'hyb_EV_pca':>11}  {'lin_EV_full':>12}  "
        f"{'lin_EV_pca':>11}  {'(h-l)_full':>11}  {'hyb_fit_s':>9}  "
        f"{'lin_fit_s':>9}  {'max_rec_s':>9}"
    )
    rows = []
    for k in ladder:
        hybrid = _fit_ev(
            z_tr,
            z_te,
            k,
            "curved",
            args.seed,
            args.manifold_iterations,
            args.linear_epochs,
            args.active_support,
            args.linear_score_mode,
            args.sep_mu,
        )
        linear = _fit_ev(
            z_tr,
            z_te,
            k,
            "linear",
            args.seed,
            args.manifold_iterations,
            args.linear_epochs,
            args.active_support,
            args.linear_score_mode,
            args.sep_mu,
        )
        ev_h_pca = hybrid["test_ev"]
        ev_l_pca = linear["test_ev"]
        # FULL-space EV = r * EV_Z (see _pca_project): the PCA-space EV is inflated
        # by ~1/r relative to a full-space reconstruction, so the acceptance gate and
        # the official-reference parity must use the full-space numbers.
        ev_h_full = r * ev_h_pca
        ev_l_full = r * ev_l_pca
        print(
            f"{k:>4}  {ev_h_full:>12.6f}  {ev_h_pca:>11.6f}  {ev_l_full:>12.6f}  "
            f"{ev_l_pca:>11.6f}  {ev_h_full - ev_l_full:>11.6f}  "
            f"{hybrid['fit_seconds']:>8.1f}  {linear['fit_seconds']:>8.1f}  "
            f"{max(hybrid['reconstruct_seconds'], linear['reconstruct_seconds']):>8.1f}"
        )
        comparison_contract = (
            "dense_same_model_family"
            if k <= realized_p
            else "overcomplete_matched_k_data_metric_and_exact_support"
        )
        rows.append(
            {
                "K": k,
                "P": realized_p,
                "comparison_contract": comparison_contract,
                "configured_active_support": int(args.active_support),
                "overcomplete_active_support": (
                    int(args.active_support) if k > realized_p else None
                ),
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
                "hybrid_fit_seconds": hybrid["fit_seconds"],
                "linear_fit_seconds": linear["fit_seconds"],
                "max_reconstruct_seconds": max(
                    hybrid["reconstruct_seconds"], linear["reconstruct_seconds"]
                ),
                "hybrid_route": hybrid["route"],
                "linear_route": linear["route"],
                "hybrid_support_evidence": hybrid["support_evidence"],
                "linear_support_evidence": linear["support_evidence"],
                "hybrid_certificate": hybrid["certificate"],
                "linear_certificate": linear["certificate"],
                "linear_score_route_stats": linear.get("score_route_stats"),
                "hybrid_seed_topology": "circle",
                "hybrid_atom_topologies": hybrid["atom_topologies"],
                "hybrid_split": hybrid["hybrid_split"],
                "linear_atom_topologies": linear["atom_topologies"],
                "linear_split": linear["hybrid_split"],
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
            "realized_p": realized_p,
            "pca_retained_var_ratio": float(r),
            "max_rows": args.max_rows,
            "seed": int(args.seed),
            "active_support": int(args.active_support),
            "manifold_iterations": int(args.manifold_iterations),
            "linear_epochs": int(args.linear_epochs),
            "linear_score_mode": args.linear_score_mode,
            "cpu_partial": bool(args.cpu_partial),
            "acceptance": acceptance,
            "rows": rows,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"JSON -> {args.out}")


if __name__ == "__main__":
    main()
