#!/usr/bin/env python3
"""Gemma geometric-wall closure rerun with real GAM block-chart promotion.

This driver harvests the exact Gemma residual layers, records per-token positions,
runs the nuisance-atlas position-0 peel, then compares a linear GAM block-SAE
baseline against GAM's real curved block-chart promotion lane at exactly matched
parameter count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wall_closure_common import (
    build_strata,
    correlation,
    fit_stratum,
    load_layer_with_positions,
    log,
    matched_curved_blocks,
    position0_nuisance_peel,
    write_json,
)


def parse_layers(raw: str) -> list[int]:
    layers = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not layers:
        raise SystemExit("at least one layer is required")
    return layers


def load_texts(args: argparse.Namespace) -> list[str]:
    from datasets import load_dataset

    texts: list[str] = []
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    for row in ds:
        text = (row.get(args.text_field) or row.get("content") or "").strip()
        if len(text) < args.min_chars:
            continue
        texts.append(text)
        if len(texts) >= args.max_docs:
            break
    if not texts:
        raise SystemExit("dataset stream produced no usable texts")
    return texts


def hidden_size_from_model(model: Any) -> int:
    config = model.config
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    return int(config.text_config.hidden_size)


def harvest(args: argparse.Namespace, layers: list[int], out_dir: Path) -> tuple[dict[int, Path], Path]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    paths = {layer: out_dir / f"acts_L{layer}.npy" for layer in layers}
    pos_path = out_dir / "positions.npy"
    if all(path.exists() for path in paths.values()) and pos_path.exists():
        log("using cached activation arrays")
        return paths, pos_path

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    d_model = hidden_size_from_model(model)
    arrays = {
        layer: np.lib.format.open_memmap(
            paths[layer], mode="w+", dtype=np.float32, shape=(args.n_tokens, d_model)
        )
        for layer in layers
    }
    positions = np.lib.format.open_memmap(
        pos_path, mode="w+", dtype=np.int32, shape=(args.n_tokens,)
    )

    texts = load_texts(args)
    cursor = 0
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_docs):
            batch = texts[start : start + args.batch_docs]
            encoded = tokenizer(
                batch,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            lengths = encoded["attention_mask"].sum(dim=1).cpu().tolist()
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=True, use_cache=False)
            take_remaining = args.n_tokens - cursor
            for row, length in enumerate(lengths):
                if take_remaining <= 0:
                    break
                take = min(int(length), take_remaining)
                positions[cursor : cursor + take] = np.arange(take, dtype=np.int32)
                for layer in layers:
                    hidden = output.hidden_states[layer + 1][row, :take, :].detach().float().cpu()
                    arrays[layer][cursor : cursor + take] = hidden.numpy()
                cursor += take
                take_remaining = args.n_tokens - cursor
            log(f"harvested {cursor}/{args.n_tokens} residual rows")
            del output, encoded
            if cursor >= args.n_tokens:
                break
    if cursor < args.n_tokens:
        raise SystemExit(f"only harvested {cursor} tokens; increase --max-docs")
    for arr in arrays.values():
        arr.flush()
    positions.flush()
    return paths, pos_path


def fit_layer(
    args: argparse.Namespace,
    layer: int,
    act_path: Path,
    positions_path: Path,
) -> dict[str, Any]:
    label = f"gemma_L{layer}"
    log(f"{label}: loading residuals and positions")
    sampled = load_layer_with_positions(act_path, positions_path)
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
            fitted.append(fit_stratum(label, spec, peeled, args.flat_blocks, curved_blocks, args))
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
        "layer": int(layer),
        "label": label,
        "path": str(act_path),
        "positions_path": str(positions_path),
        "source_shape": sampled.source_shape,
        "n_tokens": int(sampled.x.shape[0]),
        "d_model": int(sampled.x.shape[1]),
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
        "# Gemma Geometric-Wall Closure Rerun",
        "",
        "This is the decisive rerun design, not the matched-quadratic proxy.",
        "The harvested Gemma residual layers are residualized against the nuisance-atlas position-0 atom, then fit with GAM's real block-chart promotion lane.",
        "",
        "## Summary",
        "",
        f"- model: `{payload['model']}`",
        f"- tokens per layer: {payload['n_tokens']}",
        f"- layers: {', '.join(str(x['layer']) for x in payload['layers'])}",
        "",
        "## Pooled Layer Floors",
        "",
        "| layer | pos0 absorbed | fitted strata | rows | flat params | curved params | flat floor | curved floor | drop | curvature/drop r |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--layers", default="12,25")
    ap.add_argument("--n-tokens", type=int, default=30_000)
    ap.add_argument("--max-docs", type=int, default=2000)
    ap.add_argument("--batch-docs", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-position0-rows", type=int, default=8)
    ap.add_argument("--min-stratum-rows", type=int, default=512)
    ap.add_argument("--flat-blocks", type=int, default=24)
    ap.add_argument("--curved-blocks", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--chart-basis", type=int, default=4)
    ap.add_argument("--block-topk", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=8)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--block-tile", type=int, default=512)
    ap.add_argument("--frame-ridge", type=float, default=1.0e-9)
    ap.add_argument("--tolerance", type=float, default=1.0e-5)
    ap.add_argument("--min-firings", type=int, default=32)
    ap.add_argument("--max-chart-blocks", type=int, default=256)
    ap.add_argument("--crossfit-folds", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--whitening-ridge", type=float, default=1.0e-8)
    ap.add_argument("--pair-screen", action="store_true")
    ap.add_argument("--pair-top-blocks", type=int, default=64)
    ap.add_argument("--max-pairs", type=int, default=128)
    ap.add_argument("--pair-min-cofirings", type=int, default=64)
    ap.add_argument("--pair-min-score", type=float, default=0.20)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)
    act_paths, positions_path = harvest(args, layers, args.out_dir)
    results = [fit_layer(args, layer, act_paths[layer], positions_path) for layer in layers]
    settings = {
        "min_position0_rows": int(args.min_position0_rows),
        "min_stratum_rows": int(args.min_stratum_rows),
        "flat_blocks": int(args.flat_blocks),
        "curved_blocks": int(results[0]["curved_blocks"]),
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
        "experiment": "gemma_wall_closure_real_block_chart_post_pos0_peel",
        "engine": "gamfit.block_sparse_dictionary_fit + BlockSparseDictionaryFit.compose_block_charts",
        "nuisance_peel": "OLS nuisance atlas design [intercept, position0_indicator]",
        "model": args.model,
        "n_tokens": args.n_tokens,
        "settings": settings,
        "layers": results,
    }
    write_json(args.out_dir / "numbers.json", payload)
    write_report(payload, args.out_dir)
    print("RESULTS_JSON " + json.dumps(payload))


if __name__ == "__main__":
    main()
