from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import gamfit


@dataclass(frozen=True)
class LayerSpec:
    label: str
    path: Path


@dataclass(frozen=True)
class SampledLayer:
    x: np.ndarray
    positions: np.ndarray
    source_shape: list[int]


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


def parse_layer_specs(raw_specs: list[str]) -> list[LayerSpec]:
    specs: list[LayerSpec] = []
    for raw in raw_specs:
        if ":" not in raw:
            raise SystemExit(f"layer spec must be LABEL:PATH, got {raw!r}")
        label, path = raw.split(":", 1)
        label = label.strip()
        if not label:
            raise SystemExit(f"empty layer label in {raw!r}")
        specs.append(LayerSpec(label=label, path=Path(path)))
    if not specs:
        raise SystemExit("at least one --layer is required")
    return specs


def parse_position_specs(raw_specs: list[str]) -> dict[str, Path]:
    positions: dict[str, Path] = {}
    for raw in raw_specs:
        if ":" not in raw:
            raise SystemExit(f"position spec must be LABEL:PATH, got {raw!r}")
        label, path = raw.split(":", 1)
        label = label.strip()
        if not label:
            raise SystemExit(f"empty position label in {raw!r}")
        if label in positions:
            raise SystemExit(f"duplicate --positions label {label!r}")
        positions[label] = Path(path)
    return positions


def sample_rows_with_positions(layer: LayerSpec, positions_path: Path, n_rows: int, seed: int) -> SampledLayer:
    arr = np.load(layer.path, mmap_mode="r")
    if arr.ndim != 2:
        raise SystemExit(f"{layer.path} must be rank-2, got shape {arr.shape}")
    positions = np.load(positions_path, mmap_mode="r")
    if positions.ndim != 1:
        raise SystemExit(f"{positions_path} must be rank-1, got shape {positions.shape}")
    if int(positions.shape[0]) < int(arr.shape[0]):
        raise SystemExit(
            f"{positions_path} has {positions.shape[0]} rows but {layer.path} has {arr.shape[0]}"
        )
    take = min(n_rows, int(arr.shape[0]))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(arr.shape[0], size=take, replace=False))
    x = np.asarray(arr[indices], dtype=np.float32)
    pos = np.asarray(positions[indices], dtype=np.int32)
    return SampledLayer(
        x=np.ascontiguousarray(x),
        positions=np.ascontiguousarray(pos),
        source_shape=[int(arr.shape[0]), int(arr.shape[1])],
    )


def load_layer_with_positions(layer_path: Path, positions_path: Path) -> SampledLayer:
    x = np.asarray(np.load(layer_path, mmap_mode="r"), dtype=np.float32)
    positions = np.asarray(np.load(positions_path, mmap_mode="r"), dtype=np.int32)
    if x.ndim != 2:
        raise SystemExit(f"{layer_path} must be rank-2, got shape {x.shape}")
    if positions.ndim != 1:
        raise SystemExit(f"{positions_path} must be rank-1, got shape {positions.shape}")
    if int(positions.shape[0]) != int(x.shape[0]):
        raise SystemExit(f"{positions_path} has {positions.shape[0]} rows but {layer_path} has {x.shape[0]}")
    return SampledLayer(
        x=np.ascontiguousarray(x),
        positions=np.ascontiguousarray(positions),
        source_shape=[int(x.shape[0]), int(x.shape[1])],
    )


def squared_energy(x: np.ndarray) -> float:
    x64 = x.astype(np.float64, copy=False)
    return float(np.sum(x64 * x64))


def centered_energy(x: np.ndarray) -> float:
    x64 = x.astype(np.float64, copy=False)
    xc = x64 - np.mean(x64, axis=0, keepdims=True)
    return float(np.sum(xc * xc))


def position0_nuisance_peel(
    x: np.ndarray,
    positions: np.ndarray,
    min_position0_rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    pos0 = positions == 0
    n_pos0 = int(np.count_nonzero(pos0))
    n_other = int(positions.shape[0] - n_pos0)
    if n_pos0 < min_position0_rows:
        raise SystemExit(f"position-0 nuisance peel needs at least {min_position0_rows} pos0 rows, got {n_pos0}")
    if n_other < min_position0_rows:
        raise SystemExit(f"position-0 nuisance peel needs non-pos0 rows, got {n_other}")
    residual64 = x.astype(np.float64, copy=True)
    other_mean = np.mean(residual64[~pos0], axis=0)
    pos0_mean = np.mean(residual64[pos0], axis=0)
    pos0_atom = pos0_mean - other_mean
    residual64[~pos0] -= other_mean
    residual64[pos0] -= pos0_mean
    residual = np.ascontiguousarray(residual64.astype(np.float32))
    total = centered_energy(x)
    residual_energy = squared_energy(residual)
    return residual, {
        "n_rows": int(x.shape[0]),
        "position0_rows": n_pos0,
        "non_position0_rows": n_other,
        "design_columns": 2,
        "absorbed_centered_fraction": float(1.0 - residual_energy / max(total, 1.0e-30)),
        "position0_atom_energy_fraction": float(
            n_pos0 * float(np.dot(pos0_atom, pos0_atom)) / max(total, 1.0e-30)
        ),
    }


def energy_floor(target: np.ndarray, prediction: np.ndarray) -> float:
    return squared_energy(target - prediction) / max(squared_energy(target), 1.0e-30)


def exponent_bins(energies: np.ndarray) -> np.ndarray:
    out = np.zeros(energies.shape[0], dtype=np.int32)
    mask = np.isfinite(energies) & (energies > 0.0)
    exponent = np.frexp(energies[mask])[1]
    if exponent.shape[0] > 0:
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
        specs.insert(
            0,
            {
                "exp_lo": int(a["exp_lo"]),
                "exp_hi": int(b["exp_hi"]),
                "rows": rows,
                "n_rows": int(rows.shape[0]),
                "mean_energy": float(np.mean(vals)),
                "std_energy": float(np.std(vals)),
            },
        )
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


def matched_curved_blocks(flat_blocks: int, block_size: int, chart_basis: int, requested: int) -> int:
    flat_units = flat_blocks * block_size
    curved_unit = block_size + chart_basis
    if requested > 0:
        curved_blocks = requested
    else:
        if flat_units % curved_unit != 0:
            raise SystemExit(
                "flat_blocks * block_size must be divisible by block_size + chart_basis "
                "for exact parameter matching"
            )
        curved_blocks = flat_units // curved_unit
    if curved_blocks * curved_unit != flat_units:
        raise SystemExit(
            f"parameter mismatch: flat units={flat_units}, curved units={curved_blocks * curved_unit}"
        )
    return int(curved_blocks)


def fit_block_dictionary(x: np.ndarray, n_blocks: int, args: Any) -> Any:
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
            out[key] = value.item() if isinstance(value, np.generic) else value
        rows.append(out)
    return rows


def fit_stratum(
    layer_label: str,
    spec: StratumSpec,
    x: np.ndarray,
    flat_blocks: int,
    curved_blocks: int,
    args: Any,
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
        pair_screen=bool(args.pair_screen),
        pair_top_blocks=int(args.pair_top_blocks),
        max_pairs=int(args.max_pairs),
        pair_min_cofirings=int(args.pair_min_cofirings),
        pair_min_score=float(args.pair_min_score),
    )
    curved_recon = np.ascontiguousarray(composed["reconstructed"], dtype=np.float32)
    curved_floor = energy_floor(xs, curved_recon)
    correction = curved_recon - curved.fitted
    curvature_proxy = math.sqrt(squared_energy(correction) / max(target_energy, 1.0e-30))
    block_records = chart_records(composed.get("blocks", []))
    pair_records = chart_records(composed.get("pairs", []))
    accepted_blocks = [int(v) for v in composed.get("accepted_blocks", [])]
    accepted_pairs = [list(map(int, v)) for v in composed.get("accepted_pairs", [])]
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
        "accepted_pairs": accepted_pairs,
        "n_chart_records": int(len(block_records) + len(pair_records)),
        "n_accepted_charts": int(len(accepted_blocks) + len(accepted_pairs)),
        "block_chart_records": block_records,
        "pair_chart_records": pair_records,
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
