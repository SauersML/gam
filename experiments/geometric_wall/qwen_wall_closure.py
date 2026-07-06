#!/usr/bin/env python3
"""Qwen geometric-wall closure rerun with real GAM block-chart promotion.

This driver consumes pre-harvested Qwen residual arrays and order-matched
within-document position arrays. It first runs the nuisance-atlas position-0
peel, then compares a linear GAM block-SAE baseline against GAM's real curved
block-chart promotion lane at exactly matched parameter count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wall_closure_common import (
    LayerSpec,
    build_strata,
    correlation,
    fit_stratum,
    log,
    matched_curved_blocks,
    parse_layer_specs,
    parse_position_specs,
    position0_nuisance_peel,
    sample_rows_with_positions,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", action="append", required=True, help="LABEL:PATH residual .npy")
    parser.add_argument(
        "--positions",
        action="append",
        required=True,
        help="LABEL:PATH int position .npy, with labels matching --layer",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-rows", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--min-position0-rows", type=int, default=8)
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
    parser.add_argument("--pair-screen", action="store_true")
    parser.add_argument("--pair-top-blocks", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=128)
    parser.add_argument("--pair-min-cofirings", type=int, default=64)
    parser.add_argument("--pair-min-score", type=float, default=0.20)
    return parser.parse_args()


def fit_layer(args: argparse.Namespace, layer_index: int, label: str, path: Path, positions_path: Path) -> dict[str, Any]:
    log(f"{label}: loading sampled residuals and positions")
    sampled = sample_rows_with_positions(
        layer=LayerSpec(label=label, path=path),
        positions_path=positions_path,
        n_rows=args.sample_rows,
        seed=args.seed + 1009 * layer_index,
    )
    peeled, peel_stats = position0_nuisance_peel(
        sampled.x,
        sampled.positions,
        min_position0_rows=args.min_position0_rows,
    )
    log(
        f"{label}: position-0 nuisance absorbed "
        f"{peel_stats['absorbed_centered_fraction']:.6f} of centered energy"
    )
    strata = build_strata(peeled)
    curved_blocks = matched_curved_blocks(
        args.flat_blocks,
        args.block_size,
        args.chart_basis,
        args.curved_blocks,
    )
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
            fitted.append(
                fit_stratum(label, spec, peeled, args.flat_blocks, curved_blocks, args)
            )
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
    if not fitted:
        raise SystemExit(f"{label}: no strata were fitted")
    flat_rss = sum(row["flat_floor"] * row["target_energy"] for row in fitted)
    curved_rss = sum(row["curved_floor"] * row["target_energy"] for row in fitted)
    total_energy = sum(row["target_energy"] for row in fitted)
    drops = [float(row["drop"]) for row in fitted]
    curvatures = [float(row["curvature_proxy"]) for row in fitted]
    return {
        "label": label,
        "path": str(path),
        "positions_path": str(positions_path),
        "source_shape": sampled.source_shape,
        "sample_rows": int(sampled.x.shape[0]),
        "dimension": int(sampled.x.shape[1]),
        "nuisance_peel": peel_stats,
        "strata_total": int(len(strata)),
        "strata_fitted": int(len(fitted)),
        "strata_skipped": skipped,
        "covered_rows": int(sum(row["n_rows"] for row in fitted)),
        "flat_blocks": int(args.flat_blocks),
        "curved_blocks": int(curved_blocks),
        "block_size": int(args.block_size),
        "chart_basis": int(args.chart_basis),
        "flat_total_params": int(args.flat_blocks * args.block_size * sampled.x.shape[1]),
        "curved_total_params": int(curved_blocks * (args.block_size + args.chart_basis) * sampled.x.shape[1]),
        "pooled_flat_floor": float(flat_rss / max(total_energy, 1.0e-30)),
        "pooled_curved_floor": float(curved_rss / max(total_energy, 1.0e-30)),
        "pooled_drop": float((flat_rss - curved_rss) / max(total_energy, 1.0e-30)),
        "mean_flat_floor": float(np.mean([row["flat_floor"] for row in fitted])),
        "mean_curved_floor": float(np.mean([row["curved_floor"] for row in fitted])),
        "mean_drop": float(np.mean(drops)),
        "mean_curvature_proxy": float(np.mean(curvatures)),
        "curvature_drop_correlation": correlation(curvatures, drops),
        "strata": fitted,
    }


def write_report(payload: dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# Qwen Geometric-Wall Closure Rerun",
        "",
        "This is the decisive rerun design, not the matched-quadratic proxy.",
        "Each Qwen layer is sampled with its order-matched positions, residualized against the nuisance-atlas position-0 atom, then fit with GAM's real block-chart promotion lane.",
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
            "| layer | pos0 absorbed | fitted strata | rows | flat params | curved params | flat floor | curved floor | drop | curvature/drop r |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for layer in payload["layers"]:
        corr = layer["curvature_drop_correlation"]
        corr_s = "NA" if corr is None else f"{corr:.6f}"
        lines.append(
            "| {label} | {absorbed:.6f} | {strata_fitted}/{strata_total} | {covered_rows} | "
            "{flat_total_params} | {curved_total_params} | {pooled_flat_floor:.6f} | "
            "{pooled_curved_floor:.6f} | {pooled_drop:.6f} | {corr} |".format(
                **layer,
                absorbed=layer["nuisance_peel"]["absorbed_centered_fraction"],
                corr=corr_s,
            )
        )
    lines.extend(
        [
            "",
            "## Decisive Prediction",
            "",
            "- Curved-residual theory is confirmed if the post-peel curved floor is below the matched linear floor, with positive pooled drop and larger drops in strata with larger accepted chart-correction energy.",
            "- Density-only theory is supported if the post-peel curved and linear floors are statistically indistinguishable, or if curved remains higher, despite accepted chart opportunities.",
            "- The expected curved-residual scale is a floor drop proportional to kappa^2 * ell^4 / tile; the table reports the empirical drop and the RMS chart-correction proxy needed to check that ordering.",
        ]
    )
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    specs = parse_layer_specs(args.layer)
    position_specs = parse_position_specs(args.positions)
    missing = sorted(spec.label for spec in specs if spec.label not in position_specs)
    if missing:
        raise SystemExit(f"missing --positions for labels: {', '.join(missing)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    layers = [
        fit_layer(args, i, spec.label, spec.path, position_specs[spec.label])
        for i, spec in enumerate(specs)
    ]
    settings = {
        "sample_rows": int(args.sample_rows),
        "seed": int(args.seed),
        "min_position0_rows": int(args.min_position0_rows),
        "min_stratum_rows": int(args.min_stratum_rows),
        "flat_blocks": int(args.flat_blocks),
        "curved_blocks": int(layers[0]["curved_blocks"]),
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
        "pair_screen": bool(args.pair_screen),
    }
    payload = {
        "experiment": "qwen_wall_closure_real_block_chart_post_pos0_peel",
        "engine": "gamfit.block_sparse_dictionary_fit + BlockSparseDictionaryFit.compose_block_charts",
        "nuisance_peel": "OLS nuisance atlas design [intercept, position0_indicator]",
        "settings": settings,
        "layers": layers,
    }
    write_json(args.out_dir / "numbers.json", payload)
    write_report(payload, args.out_dir)
    print("RESULTS_JSON " + json.dumps(payload))


if __name__ == "__main__":
    main()
