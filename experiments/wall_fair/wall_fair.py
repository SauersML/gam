#!/usr/bin/env python3
"""Fair wall-closure replication with Rust block dictionaries.

The driver runs on MSI against a locally built ``gamfit`` wheel. It compares a
flat block-sparse dictionary with a curved block-coordinate chart composition at
matched parameter count, after a Qwen position-0 sink peel and before pooling
across residual-energy strata.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import gamfit


@dataclass(frozen=True)
class StratumSpec:
    index: int
    exp_lo: int
    exp_hi: int
    n_rows: int
    mean_energy: float
    std_energy: float
    rows: np.ndarray


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", default=["18", "30"])
    parser.add_argument("--sample-rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--pca-rank", type=int, default=8)
    parser.add_argument("--pca-power-iter", type=int, default=1)
    parser.add_argument("--min-stratum-rows", type=int, default=512)
    parser.add_argument("--flat-blocks", type=int, default=24)
    parser.add_argument("--curved-blocks", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--chart-basis", type=int, default=4)
    parser.add_argument("--block-topk", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--block-tile", type=int, default=512)
    parser.add_argument("--frame-ridge", type=float, default=1.0e-9)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--min-firings", type=int, default=32)
    parser.add_argument("--max-chart-blocks", type=int, default=256)
    parser.add_argument("--crossfit-folds", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--whitening-ridge", type=float, default=1.0e-8)
    return parser.parse_args()


def sample_rows(path: Path, n_rows: int, seed: int) -> tuple[np.ndarray, list[int]]:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"{path} must be rank-2, got shape {arr.shape}")
    take = min(n_rows, int(arr.shape[0]))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(arr.shape[0], size=take, replace=False))
    x = np.asarray(arr[indices], dtype=np.float32)
    return np.ascontiguousarray(x), [int(arr.shape[0]), int(arr.shape[1])]


def randomized_top_pc(
    x: np.ndarray, rank: int, power_iter: int, seed: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = x.astype(np.float32, copy=True)
    centered -= mean
    total_energy = squared_energy(centered)
    sketch_rank = min(max(rank, 2), centered.shape[0] - 1, centered.shape[1])
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((centered.shape[1], sketch_rank)).astype(np.float32)
    y = centered @ omega
    for _iteration in range(power_iter):
        y = centered @ (centered.T @ y)
    q, _r = np.linalg.qr(y, mode="reduced")
    b = q.T @ centered
    _u, singular_values, vt = np.linalg.svd(b, full_matrices=False)
    top = vt[0].astype(np.float32)
    sink_energy = float(singular_values[0] * singular_values[0])
    return mean, top, sink_energy, total_energy


def peel_sink(x: np.ndarray, seed: int, rank: int, power_iter: int) -> tuple[np.ndarray, float]:
    mean, sink, sink_energy, total_energy = randomized_top_pc(x, rank, power_iter, seed)
    centered = x.astype(np.float32, copy=True)
    centered -= mean
    scores = centered @ sink
    centered -= scores[:, None] * sink[None, :]
    sink_fraction = sink_energy / max(total_energy, 1.0e-30)
    return np.ascontiguousarray(centered), float(sink_fraction)


def squared_energy(x: np.ndarray) -> float:
    x64 = x.astype(np.float64, copy=False)
    return float(np.sum(x64 * x64))


def energy_floor(target: np.ndarray, prediction: np.ndarray) -> float:
    return squared_energy(target - prediction) / max(squared_energy(target), 1.0e-30)


def exponent_bins(energies: np.ndarray) -> np.ndarray:
    out = np.zeros(energies.shape[0], dtype=np.int32)
    mask = np.isfinite(energies) & (energies > 0.0)
    mantissa, exponent = np.frexp(energies[mask])
    if mantissa.shape[0] > 0:
        out[mask] = exponent.astype(np.int32) - 1 + 1023
    return out


def sturges_cap(n_rows: int) -> int:
    if n_rows <= 1:
        return 1
    return int(math.floor(math.log2(n_rows))) + 1


def build_strata(x: np.ndarray) -> list[StratumSpec]:
    energies = np.sum(x.astype(np.float64) * x.astype(np.float64), axis=1)
    bins = exponent_bins(energies)
    specs: list[dict[str, Any]] = []
    for exp in sorted(int(v) for v in np.unique(bins)):
        rows = np.flatnonzero(bins == exp).astype(np.int64)
        vals = energies[rows]
        specs.append(
            {
                "exp_lo": exp,
                "exp_hi": exp,
                "rows": rows,
                "n_rows": int(rows.shape[0]),
                "mean_energy": float(np.mean(vals)),
                "std_energy": float(np.std(vals)),
            }
        )
    cap = sturges_cap(x.shape[0])
    while len(specs) > cap and len(specs) >= 2:
        a = specs.pop(0)
        b = specs.pop(0)
        rows = np.concatenate([a["rows"], b["rows"]])
        vals = energies[rows]
        merged = {
            "exp_lo": int(a["exp_lo"]),
            "exp_hi": int(b["exp_hi"]),
            "rows": rows,
            "n_rows": int(rows.shape[0]),
            "mean_energy": float(np.mean(vals)),
            "std_energy": float(np.std(vals)),
        }
        specs.insert(0, merged)
    return [
        StratumSpec(
            index=i,
            exp_lo=int(spec["exp_lo"]),
            exp_hi=int(spec["exp_hi"]),
            n_rows=int(spec["n_rows"]),
            mean_energy=float(spec["mean_energy"]),
            std_energy=float(spec["std_energy"]),
            rows=np.asarray(spec["rows"], dtype=np.int64),
        )
        for i, spec in enumerate(specs)
    ]


def fit_block_dictionary(x: np.ndarray, n_blocks: int, args: argparse.Namespace) -> Any:
    return gamfit.block_sparse_dictionary_fit(
        np.ascontiguousarray(x, dtype=np.float32),
        int(n_blocks),
        block_size=int(args.block_size),
        block_topk=int(args.block_topk),
        max_epochs=int(args.max_epochs),
        minibatch=int(args.minibatch),
        block_tile=int(args.block_tile),
        frame_ridge=float(args.frame_ridge),
        aux_k=max(1, int(n_blocks) // 8),
        tolerance=float(args.tolerance),
    )


def chart_records(records: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in records:
        data = dict(row)
        out: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, np.generic):
                out[key] = value.item()
            else:
                out[key] = value
        rows.append(out)
    return rows


def fit_stratum(
    layer_label: str,
    spec: StratumSpec,
    x: np.ndarray,
    flat_blocks: int,
    curved_blocks: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    xs = np.ascontiguousarray(x[spec.rows], dtype=np.float32)
    target_energy = squared_energy(xs)
    log(
        f"{layer_label} stratum {spec.index}: rows={xs.shape[0]} "
        f"flat_blocks={flat_blocks} curved_blocks={curved_blocks}"
    )
    flat = fit_block_dictionary(xs, flat_blocks, args)
    flat_floor = energy_floor(xs, flat.fitted)
    curved = fit_block_dictionary(xs, curved_blocks, args)
    composed = curved.compose_block_charts(
        xs,
        residual_target=True,
        min_firings=int(args.min_firings),
        max_blocks=int(args.max_chart_blocks),
        crossfit_folds=int(args.crossfit_folds),
        alpha=float(args.alpha),
        min_effect=0.0,
        whitening_ridge=float(args.whitening_ridge),
        pair_screen=False,
    )
    curved_recon = np.ascontiguousarray(composed["reconstructed"], dtype=np.float32)
    curved_floor = energy_floor(xs, curved_recon)
    correction = curved_recon - curved.fitted
    curvature_proxy = math.sqrt(squared_energy(correction) / max(target_energy, 1.0e-30))
    block_records = chart_records(composed.get("blocks", []))
    accepted_blocks = [int(v) for v in composed.get("accepted_blocks", [])]
    return {
        "stratum": int(spec.index),
        "exp_lo": int(spec.exp_lo),
        "exp_hi": int(spec.exp_hi),
        "unbiased_exp_lo": int(spec.exp_lo - 1023) if spec.exp_lo > 0 else None,
        "unbiased_exp_hi": int(spec.exp_hi - 1023) if spec.exp_hi > 0 else None,
        "n_rows": int(xs.shape[0]),
        "mean_energy": float(spec.mean_energy),
        "std_energy": float(spec.std_energy),
        "target_energy": float(target_energy),
        "flat_blocks": int(flat_blocks),
        "curved_blocks": int(curved_blocks),
        "flat_params": int(flat_blocks * args.block_size * xs.shape[1]),
        "curved_params": int(curved_blocks * (args.block_size + args.chart_basis) * xs.shape[1]),
        "flat_floor": float(flat_floor),
        "curved_floor": float(curved_floor),
        "drop": float(flat_floor - curved_floor),
        "curvature_proxy": float(curvature_proxy),
        "flat_ev": float(flat.explained_variance),
        "curved_base_ev": float(curved.explained_variance),
        "flat_epochs": int(flat.epochs),
        "curved_epochs": int(curved.epochs),
        "flat_converged": bool(flat.converged),
        "curved_converged": bool(curved.converged),
        "selected_blocks": [int(v) for v in composed.get("selected_blocks", [])],
        "accepted_blocks": accepted_blocks,
        "n_chart_records": int(len(block_records)),
        "n_accepted_charts": int(len(accepted_blocks)),
        "chart_records": block_records,
    }


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return None
    return float(np.dot(x, y) / denom)


def fit_layer(layer: str, args: argparse.Namespace) -> dict[str, Any]:
    label = f"qwen3_8b_L{layer}"
    path = args.data_root / "harvest_out" / "qwen3_8b_wikitext" / f"resid_L{layer}.npy"
    log(f"{label}: loading {path}")
    raw, source_shape = sample_rows(path, args.sample_rows, args.seed + int(layer))
    x, sink_fraction = peel_sink(raw, args.seed + 10_000 + int(layer), args.pca_rank, args.pca_power_iter)
    strata = build_strata(x)
    flat_blocks = int(args.flat_blocks)
    if args.curved_blocks > 0:
        curved_blocks = int(args.curved_blocks)
    else:
        curved_blocks = max(1, (flat_blocks * args.block_size) // (args.block_size + args.chart_basis))
    fitted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for spec in strata:
        if spec.n_rows < args.min_stratum_rows:
            skipped.append(
                {
                    "stratum": int(spec.index),
                    "n_rows": int(spec.n_rows),
                    "exp_lo": int(spec.exp_lo),
                    "exp_hi": int(spec.exp_hi),
                    "reason": "below_min_stratum_rows",
                }
            )
            continue
        try:
            fitted.append(fit_stratum(label, spec, x, flat_blocks, curved_blocks, args))
        except Exception as exc:
            skipped.append(
                {
                    "stratum": int(spec.index),
                    "n_rows": int(spec.n_rows),
                    "exp_lo": int(spec.exp_lo),
                    "exp_hi": int(spec.exp_hi),
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
    flat_rss = sum(row["flat_floor"] * row["target_energy"] for row in fitted)
    curved_rss = sum(row["curved_floor"] * row["target_energy"] for row in fitted)
    total_energy = sum(row["target_energy"] for row in fitted)
    drops = [float(row["drop"]) for row in fitted]
    curvatures = [float(row["curvature_proxy"]) for row in fitted]
    return {
        "label": label,
        "path": str(path),
        "source_shape": source_shape,
        "sample_rows": int(raw.shape[0]),
        "dimension": int(raw.shape[1]),
        "sink_top_pc_fraction": float(sink_fraction),
        "strata_total": int(len(strata)),
        "strata_fitted": int(len(fitted)),
        "strata_skipped": skipped,
        "covered_rows": int(sum(row["n_rows"] for row in fitted)),
        "flat_blocks": flat_blocks,
        "curved_blocks": curved_blocks,
        "block_size": int(args.block_size),
        "chart_basis": int(args.chart_basis),
        "flat_total_params": int(flat_blocks * args.block_size * raw.shape[1]),
        "curved_total_params": int(curved_blocks * (args.block_size + args.chart_basis) * raw.shape[1]),
        "pooled_flat_floor": float(flat_rss / max(total_energy, 1.0e-30)),
        "pooled_curved_floor": float(curved_rss / max(total_energy, 1.0e-30)),
        "pooled_drop": float((flat_rss - curved_rss) / max(total_energy, 1.0e-30)),
        "mean_flat_floor": float(np.mean([row["flat_floor"] for row in fitted])) if fitted else None,
        "mean_curved_floor": float(np.mean([row["curved_floor"] for row in fitted])) if fitted else None,
        "mean_drop": float(np.mean(drops)) if drops else None,
        "mean_curvature_proxy": float(np.mean(curvatures)) if curvatures else None,
        "curvature_drop_correlation": correlation(curvatures, drops),
        "strata": fitted,
    }


def write_results(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    numbers_path = output_dir / "numbers.json"
    numbers_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Fair Wall-Closure",
        "",
        "Actual Rust `gamfit` block-sparse fits, run strata-first after the Qwen sink peel.",
        "Flat and curved lanes are parameter matched by charging each curved block as "
        "`(block_size + chart_basis) * p` parameters and using fewer curved blocks.",
        "",
        "## Settings",
        "",
    ]
    for key, value in payload["settings"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Pooled Layer Floors",
            "",
            "| layer | sink frac | fitted strata | rows | flat params | curved params | flat floor | curved floor | drop | curvature/drop r |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for layer in payload["layers"]:
        corr = layer["curvature_drop_correlation"]
        corr_s = "NA" if corr is None else f"{corr:.6f}"
        lines.append(
            "| {label} | {sink_top_pc_fraction:.6f} | {strata_fitted}/{strata_total} | "
            "{covered_rows} | {flat_total_params} | {curved_total_params} | "
            "{pooled_flat_floor:.6f} | {pooled_curved_floor:.6f} | "
            "{pooled_drop:.6f} | {corr} |".format(**layer, corr=corr_s)
        )
    for layer in payload["layers"]:
        lines.extend(
            [
                "",
                f"## {layer['label']} Strata",
                "",
                "| stratum | rows | exp2 range | flat floor | curved floor | drop | curvature | accepted charts |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in layer["strata"]:
            exp_range = f"{row['unbiased_exp_lo']}..{row['unbiased_exp_hi']}"
            lines.append(
                "| {stratum} | {n_rows} | {exp} | {flat_floor:.6f} | "
                "{curved_floor:.6f} | {drop:.6f} | {curvature_proxy:.6f} | "
                "{n_accepted_charts}/{n_chart_records} |".format(**row, exp=exp_range)
            )
        if layer["strata_skipped"]:
            lines.extend(["", "Skipped strata:"])
            for skipped in layer["strata_skipped"]:
                message = skipped.get("message")
                suffix = f": {message}" if message else ""
                lines.append(
                    f"- stratum {skipped['stratum']} rows={skipped['n_rows']} "
                    f"reason={skipped['reason']}{suffix}"
                )
    lines.extend(
        [
            "",
            "Positive drop means the curved lane has a lower reconstruction floor.",
            "Curvature is the RMS chart correction energy relative to the stratum target energy.",
            "",
        ]
    )
    (output_dir / "results.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    layers = [fit_layer(layer, args) for layer in args.layers]
    settings = {
        "sample_rows": int(args.sample_rows),
        "seed": int(args.seed),
        "pca_rank": int(args.pca_rank),
        "pca_power_iter": int(args.pca_power_iter),
        "min_stratum_rows": int(args.min_stratum_rows),
        "flat_blocks": int(args.flat_blocks),
        "curved_blocks": int(args.curved_blocks),
        "block_size": int(args.block_size),
        "chart_basis": int(args.chart_basis),
        "block_topk": int(args.block_topk),
        "max_epochs": int(args.max_epochs),
        "minibatch": int(args.minibatch),
        "block_tile": int(args.block_tile),
        "frame_ridge": float(args.frame_ridge),
        "tolerance": float(args.tolerance),
        "min_firings": int(args.min_firings),
        "max_chart_blocks": int(args.max_chart_blocks),
        "crossfit_folds": int(args.crossfit_folds),
        "alpha": float(args.alpha),
        "whitening_ridge": float(args.whitening_ridge),
        "pair_screen": False,
    }
    payload = {
        "experiment": "wall_fair",
        "engine": "gamfit.block_sparse_dictionary_fit + BlockSparseDictionaryFit.compose_block_charts",
        "settings": settings,
        "layers": layers,
    }
    write_results(payload, args.output_dir)
    log(f"wrote {args.output_dir / 'numbers.json'}")
    log(f"wrote {args.output_dir / 'results.md'}")


if __name__ == "__main__":
    main()
