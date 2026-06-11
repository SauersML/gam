#!/usr/bin/env python
"""Protocol-grade concept topology race with held-out color/frame arms.

Every candidate is fit only on its arm's fit rows and all reported EV,
Gaussian log-density, and hue interpolation metrics are computed on held-out
eval rows. The output JSON carries a protocol block with deterministic split
seeds, fit budgets, capacity accounting, and the explicit nulls.
"""
from __future__ import annotations

import argparse
import colorsys
import concurrent.futures
import json
import math
import time
import traceback
from typing import Any

import numpy as np


CANDIDATES = [
    ("circle", 1),
    ("euclidean", 1),
    ("euclidean", 2),
    ("sphere", 2),
]


def load_bank_raw(bank: str, layer: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    acts = np.load(f"{bank}/activations.npy", mmap_mode="r")
    x = np.asarray(acts[:, layer, :], dtype=np.float64)
    hue = []
    rows = []
    with open(f"{bank}/prompts.jsonl") as f:
        for line in f:
            row = json.loads(line)
            r, g, b = row["rgb"]
            h, _s, _v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            hue.append(2.0 * math.pi * h)
            rows.append(row)
    hue_arr = np.asarray(hue, dtype=np.float64)
    if hue_arr.shape[0] != x.shape[0]:
        raise SystemExit(f"bank mismatch: {x.shape[0]} activations vs {hue_arr.shape[0]} prompts")
    return x, hue_arr, rows


def normalize_from_fit(x_fit_raw: np.ndarray, x_eval_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = x_fit_raw.mean(axis=0, keepdims=True)
    sd = np.maximum(x_fit_raw.std(axis=0, keepdims=True), 1e-6)
    return (x_fit_raw - mu) / sd, (x_eval_raw - mu) / sd, mu, sd


def deterministic_split(n_rows: int, eval_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < eval_fraction < 1.0:
        raise SystemExit("--eval-frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    n_eval = max(1, int(round(n_rows * eval_fraction)))
    n_eval = min(n_eval, n_rows - 1)
    return np.sort(order[n_eval:]), np.sort(order[:n_eval])


def leave_one_arms(rows: list[dict[str, Any]], field: str, prefix: str) -> list[dict[str, Any]]:
    values = sorted({str(row[field]) for row in rows if field in row})
    arms = []
    for value in values:
        eval_idx = np.asarray([i for i, row in enumerate(rows) if str(row.get(field)) == value], dtype=int)
        fit_idx = np.asarray([i for i, row in enumerate(rows) if str(row.get(field)) != value], dtype=int)
        if fit_idx.size > 0 and eval_idx.size > 0:
            arms.append({
                "name": f"{prefix}:{value}",
                "kind": f"leave_one_{field}_out",
                "field": field,
                "held_out_value": value,
                "fit_idx": fit_idx,
                "eval_idx": eval_idx,
            })
    return arms


def explained_variance(x: np.ndarray, xhat: np.ndarray) -> float:
    num = float(np.square(x - xhat).sum())
    den = float(np.square(x - x.mean(axis=0, keepdims=True)).sum())
    return 1.0 - num / den if den > 0.0 else 0.0


def fit_dispersion(x_fit: np.ndarray, xhat_fit: np.ndarray) -> float:
    return max(float(np.square(x_fit - xhat_fit).mean()), 1e-12)


def gaussian_log_density_per_scalar(x: np.ndarray, xhat: np.ndarray, dispersion: float) -> float:
    mse = float(np.square(x - xhat).mean())
    return -0.5 * (math.log(2.0 * math.pi * dispersion) + mse / dispersion)


def circular_corr(alpha: np.ndarray, beta: np.ndarray) -> float | None:
    if alpha.size < 3 or beta.size < 3:
        return None
    a = alpha - math.atan2(np.sin(alpha).mean(), np.cos(alpha).mean())
    b = beta - math.atan2(np.sin(beta).mean(), np.cos(beta).mean())
    num = float((np.sin(a) * np.sin(b)).sum())
    den = math.sqrt(float((np.sin(a) ** 2).sum()) * float((np.sin(b) ** 2).sum()))
    return num / den if den > 1e-12 else None


def circular_abs_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (pred - truth))))


def circle_phase_mapping(theta_fit: np.ndarray, hue_fit: np.ndarray,
                         theta_eval: np.ndarray) -> np.ndarray:
    candidates = []
    for sign in (1.0, -1.0):
        shifted = sign * theta_fit
        offset = math.atan2(np.sin(hue_fit - shifted).mean(), np.cos(hue_fit - shifted).mean())
        pred_fit = shifted + offset
        err = float(circular_abs_error(pred_fit, hue_fit).mean())
        candidates.append((err, sign, offset))
    _err, sign, offset = min(candidates, key=lambda item: item[0])
    return sign * theta_eval + offset


def linear_phase_mapping(coords_fit: np.ndarray, hue_fit: np.ndarray,
                         coords_eval: np.ndarray) -> np.ndarray:
    design_fit = np.column_stack([np.ones(coords_fit.shape[0]), coords_fit])
    design_eval = np.column_stack([np.ones(coords_eval.shape[0]), coords_eval])
    target = np.column_stack([np.cos(hue_fit), np.sin(hue_fit)])
    coef, *_ = np.linalg.lstsq(design_fit, target, rcond=None)
    pred = design_eval @ coef
    return np.arctan2(pred[:, 1], pred[:, 0])


def hue_interpolation_metrics(topo: str, coords_fit: np.ndarray, hue_fit: np.ndarray,
                              coords_eval: np.ndarray, hue_eval: np.ndarray) -> dict[str, Any]:
    if coords_eval.shape[0] == 0:
        return {}
    if topo == "circle":
        pred = circle_phase_mapping(coords_fit[:, 0], hue_fit, coords_eval[:, 0])
        corr = circular_corr(coords_eval[:, 0], hue_eval)
    else:
        pred = linear_phase_mapping(coords_fit, hue_fit, coords_eval)
        corr = circular_corr(pred, hue_eval)
    err = circular_abs_error(pred, hue_eval)
    return {
        "eval_hue_circular_corr": corr,
        "eval_hue_mean_abs_error_rad": float(err.mean()),
        "eval_hue_median_abs_error_rad": float(np.median(err)),
    }


def active_threshold(assignment: str, k: int) -> float:
    if assignment == "softmax":
        return 1.0 / max(1, k)
    if assignment == "jumprelu":
        return 0.0
    return 0.5


def active_channels(assignments: np.ndarray, dims: list[int], assignment: str) -> float:
    cut = active_threshold(assignment, assignments.shape[1])
    active = assignments > cut if assignment != "jumprelu" else assignments > 0.0
    charges = np.asarray([1 + int(d) for d in dims], dtype=np.float64)
    return float((active * charges.reshape(1, -1)).sum(axis=1).mean())


def parameter_count(model: Any) -> tuple[int, dict[str, int]]:
    decoder = int(sum(np.asarray(block).size for block in model.decoder_blocks))
    coords = int(sum(np.asarray(coord).size for coord in model.coords))
    assignments = int(np.asarray(model.assignments).size)
    logits = int(np.asarray(getattr(model, "low_level_logits", np.zeros((0, 0)))).size)
    total = decoder + coords + assignments + logits
    return total, {
        "decoder_coefficients": decoder,
        "fit_coordinates": coords,
        "fit_assignments": assignments,
        "fit_logits": logits,
    }


def add_eval_metrics(entry: dict[str, Any], x_eval: np.ndarray, xhat_eval: np.ndarray,
                     dispersion: float) -> None:
    entry["eval_ev"] = explained_variance(x_eval, xhat_eval)
    entry["eval_mse"] = float(np.square(x_eval - xhat_eval).mean())
    entry["fit_dispersion"] = dispersion
    entry["eval_log_density_per_scalar"] = gaussian_log_density_per_scalar(
        x_eval, xhat_eval, dispersion,
    )
    entry["eval_log_density_total"] = float(entry["eval_log_density_per_scalar"] * x_eval.size)


def worker_gamfit():
    import gamfit

    return gamfit


def fit_candidate(x_fit: np.ndarray, x_eval: np.ndarray, hue_fit: np.ndarray,
                  hue_eval: np.ndarray, topo: str, d_atom: int, n_iter: int,
                  seed: int, assignment: str) -> dict[str, Any]:
    t0 = time.time()
    entry: dict[str, Any] = {
        "method": "gam_manifold_sae",
        "role": "union_flat_patches_null" if topo == "euclidean" else "manifold_atom",
        "topology": topo,
        "d_atom": d_atom,
        "K": 1,
        "assignment": assignment,
        "n_iter": n_iter,
        "seed": seed,
    }
    try:
        gamfit = worker_gamfit()
        model = gamfit.sae_manifold_fit(
            x_fit, K=1, d_atom=d_atom, atom_topology=topo,
            assignment=assignment, n_iter=n_iter, random_state=seed,
        )
        eval_latents = model.converged_latents(x_eval)
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()[-1500:]
        entry["seconds"] = time.time() - t0
        return entry

    entry["status"] = "ok"
    entry["seconds"] = time.time() - t0
    entry["reml_score_fit"] = float(model.reml_score) if model.reml_score is not None else None
    xhat_fit = np.asarray(model.fitted, dtype=np.float64)
    xhat_eval = np.asarray(eval_latents["fitted"], dtype=np.float64)
    coords_fit = np.asarray(model.coords[0], dtype=np.float64)
    coords_eval = np.asarray(eval_latents["coords"][0], dtype=np.float64)
    assignments_eval = np.asarray(eval_latents["assignments"], dtype=np.float64)
    dims = [int(getattr(atom, "active_dim", None) or d_atom) for atom in model.atoms]
    total_params, param_breakdown = parameter_count(model)
    entry["coords_dim_fit"] = list(coords_fit.shape)
    entry["coords_dim_eval"] = list(coords_eval.shape)
    entry["active_scalar_channels_per_row"] = active_channels(assignments_eval, dims, assignment)
    entry["mean_active_atoms_per_eval_row"] = float(
        (assignments_eval > active_threshold(assignment, assignments_eval.shape[1])).sum(axis=1).mean()
    )
    entry["total_parameter_count"] = total_params
    entry["parameter_count_breakdown"] = param_breakdown
    add_eval_metrics(entry, x_eval, xhat_eval, fit_dispersion(x_fit, xhat_fit))
    entry.update(hue_interpolation_metrics(topo, coords_fit, hue_fit, coords_eval, hue_eval))
    atom = model.atoms[0]
    entry["active_dim"] = getattr(atom, "active_dim", None)
    entry["evidence"] = getattr(atom, "evidence", None)
    for field in ("residual_gauge", "metric_provenance"):
        value = getattr(model, field, None)
        if value is not None:
            try:
                json.dumps(value)
                entry[field] = value
            except TypeError:
                entry[field] = repr(value)[:4000]
    return entry


def pca_eval(x_fit: np.ndarray, x_eval: np.ndarray, rank: int) -> dict[str, Any]:
    rank = max(1, min(int(rank), min(x_fit.shape)))
    mean = x_fit.mean(axis=0, keepdims=True)
    centered_fit = x_fit - mean
    _u, _s, vt = np.linalg.svd(centered_fit, full_matrices=False)
    components = vt[:rank]
    xhat_fit = (centered_fit @ components.T) @ components + mean
    centered_eval = x_eval - mean
    xhat_eval = (centered_eval @ components.T) @ components + mean
    entry: dict[str, Any] = {
        "method": "pca_rank_matched",
        "role": "pca_rank_matched_linear_null",
        "rank": rank,
        "active_scalar_channels_per_row": float(rank),
        "total_parameter_count": int(mean.size + components.size),
    }
    add_eval_metrics(entry, x_eval, xhat_eval, fit_dispersion(x_fit, xhat_fit))
    return entry


def capacity_ranks(entries: list[dict[str, Any]], max_rank: int) -> list[int]:
    ranks = set()
    for entry in entries:
        if entry.get("status", "ok") != "ok":
            continue
        active = entry.get("active_scalar_channels_per_row")
        if active is not None:
            ranks.add(max(1, min(int(round(float(active))), max_rank)))
    return sorted(ranks)


def comparison_table(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        if entry.get("status", "ok") != "ok":
            continue
        active = entry.get("active_scalar_channels_per_row")
        if active is None:
            continue
        rows.append({
            "capacity_group_active_channels": int(round(float(active))),
            "active_scalar_channels_per_row": float(active),
            "method": entry.get("method"),
            "role": entry.get("role"),
            "topology": entry.get("topology"),
            "rank": entry.get("rank"),
            "total_parameter_count": entry.get("total_parameter_count"),
            "eval_ev": entry.get("eval_ev"),
            "eval_log_density_per_scalar": entry.get("eval_log_density_per_scalar"),
            "eval_hue_mean_abs_error_rad": entry.get("eval_hue_mean_abs_error_rad"),
        })
    return sorted(rows, key=lambda row: (
        row["capacity_group_active_channels"],
        str(row["method"]),
        str(row.get("topology")),
    ))


def run_arm(arm: dict[str, Any], x_raw: np.ndarray, hue: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    fit_idx = np.asarray(arm["fit_idx"], dtype=int)
    eval_idx = np.asarray(arm["eval_idx"], dtype=int)
    x_fit, x_eval, _mu, _sd = normalize_from_fit(x_raw[fit_idx], x_raw[eval_idx])
    hue_fit = hue[fit_idx]
    hue_eval = hue[eval_idx]
    report: dict[str, Any] = {
        "name": arm["name"],
        "kind": arm["kind"],
        "field": arm.get("field"),
        "held_out_value": arm.get("held_out_value"),
        "fit_n": int(fit_idx.size),
        "eval_n": int(eval_idx.size),
        "fit_indices": fit_idx.tolist(),
        "eval_indices": eval_idx.tolist(),
        "candidates": [],
        "pca": [],
    }
    max_workers = min(len(CANDIDATES), 4)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for topo, d in CANDIDATES:
            print(f"[fit:{arm['name']}] {topo} d={d}", flush=True)
            future = executor.submit(
                fit_candidate,
                x_fit,
                x_eval,
                hue_fit,
                hue_eval,
                topo,
                d,
                args.n_iter,
                args.seed,
                args.assignment,
            )
            futures[future] = (topo, d)
        for future in concurrent.futures.as_completed(futures):
            topo, d = futures[future]
            try:
                entry = future.result()
            except Exception as exc:
                entry = {
                    "method": "gam_manifold_sae",
                    "topology": topo,
                    "d_atom": d,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1500:],
                }
            report["candidates"].append(entry)
            print(
                f"[fit:{arm['name']}] -> {topo} d={d} status={entry['status']} "
                f"eval_ev={entry.get('eval_ev')}",
                flush=True,
            )
    max_rank = max(1, min(x_fit.shape))
    for rank in capacity_ranks(report["candidates"], max_rank):
        entry = pca_eval(x_fit, x_eval, rank)
        report["pca"].append(entry)
        print(f"[fit:{arm['name']}] pca rank={rank} eval_ev={entry['eval_ev']}", flush=True)
    report["comparison_table"] = comparison_table([*report["candidates"], *report["pca"]])
    ok = [
        c for c in report["candidates"]
        if c.get("status") == "ok" and c.get("eval_log_density_per_scalar") is not None
    ]
    if ok:
        winner = max(ok, key=lambda c: c["eval_log_density_per_scalar"])
        report["winner_eval_log_density"] = {
            "topology": winner["topology"],
            "d_atom": winner["d_atom"],
            "role": winner["role"],
            "eval_log_density_per_scalar": winner["eval_log_density_per_scalar"],
            "eval_ev": winner["eval_ev"],
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--assignment", default="ibp_map")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    split_seed = args.seed if args.split_seed is None else args.split_seed
    x_raw, hue, rows = load_bank_raw(args.bank, args.layer)
    fit_idx, eval_idx = deterministic_split(x_raw.shape[0], args.eval_frac, split_seed)
    arms = [{
        "name": "random_holdout",
        "kind": "deterministic_random_holdout",
        "fit_idx": fit_idx,
        "eval_idx": eval_idx,
    }]
    arms.extend(leave_one_arms(rows, "color", "heldout_color"))
    arms.extend(leave_one_arms(rows, "frame", "heldout_frame"))

    report: dict[str, Any] = {
        "protocol": {
            "name": "W2 color concept held-out matched-capacity topology race",
            "bank": args.bank,
            "layer": int(args.layer),
            "n_total": int(x_raw.shape[0]),
            "p": int(x_raw.shape[1]),
            "split_seed": int(split_seed),
            "eval_fraction": float(args.eval_frac),
            "arms": [
                {
                    "name": arm["name"],
                    "kind": arm["kind"],
                    "field": arm.get("field"),
                    "held_out_value": arm.get("held_out_value"),
                    "fit_n": int(np.asarray(arm["fit_idx"]).size),
                    "eval_n": int(np.asarray(arm["eval_idx"]).size),
                }
                for arm in arms
            ],
            "normalization": "mean/std fit independently on each arm's fit rows and reused for that arm's eval rows",
            "fit_budgets": {
                "seed": int(args.seed),
                "n_iter": int(args.n_iter),
                "assignment": args.assignment,
                "candidates": [{"topology": topo, "d_atom": d, "K": 1} for topo, d in CANDIDATES],
            },
            "reported_ev_log_density": "eval split only; Gaussian dispersion estimated on fit residuals",
            "capacity": "active scalar channels per eval row; manifold charges gate plus active d_k per active atom",
            "nulls": ["union_flat_patches_null: gam atom_topology=euclidean", "pca_rank_matched_linear_null"],
        },
        "arms": [],
    }
    print(
        f"[setup] X {x_raw.shape} hue range ({hue.min():.2f},{hue.max():.2f}) "
        f"arms={len(arms)}",
        flush=True,
    )

    for arm in arms:
        arm_report = run_arm(arm, x_raw, hue, args)
        report["arms"].append(arm_report)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
