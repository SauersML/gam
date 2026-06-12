#!/usr/bin/env python3
"""Layer-resolved circle-coordinate transport on the OLMo color bank.

This experiment fits a K=1 periodic atom chart to the last-token activations at
a small ladder of transformer layers, then estimates smooth circular transport
maps between ladder layers. The chart coordinate is the layer's leading PCA
phase, aligned to hue by the residual circle isometry gauge:

    theta_next = h(theta_current)

The transport map is represented by two periodic GAMs for cos(theta_next) and
sin(theta_next). The mapped angle is recovered with atan2, which keeps the
response circular while using the scalar Gaussian GAM API. Adjacent maps and
direct two-hop maps are fit on the training split; the held-out split tests the
composition law h_{l,l+2} ~= h_{l+1,l+2} o h_{l,l+1}.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
import math
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TAU = 2.0 * math.pi
DEFAULT_BANK = (
    "/home/azuser/Manifold-SAE/runs/OLMO3_32B_TRAJ_SFT/"
    "5e-5-step10790/extra"
)
DEFAULT_REPORT = "/mnt/work/exp/layer_transport_l20_l30_report.json"
DEFAULT_PROFILE_PREFIX = "/mnt/work/exp/layer_transport_l20_l30"
DEFAULT_LAYERS = list(range(20, 31))
DEFAULT_EVAL_FRACTION = 0.25


@dataclass
class TransportFit:
    layer_current: int
    layer_next: int
    cos_model: Any
    sin_model: Any
    entry: dict[str, Any]


def json_float(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def angle_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (a - b)))


def circular_abs_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.abs(angle_delta(pred, truth))


def circular_corr(alpha: np.ndarray, beta: np.ndarray) -> float:
    """Fisher-Lee circular correlation coefficient.

    Pairwise angle differences avoid the undefined circular mean that appears
    when the hue bank covers the color wheel nearly uniformly.
    """
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"circular_corr shape mismatch: {a.shape} vs {b.shape}")
    i, j = np.triu_indices(a.shape[0], k=1)
    sa = np.sin(a[i] - a[j])
    sb = np.sin(b[i] - b[j])
    num = float((sa * sb).sum())
    den = math.sqrt(float((sa * sa).sum()) * float((sb * sb).sum()))
    return num / den if den > 0.0 else 0.0


def load_bank(bank: Path) -> tuple[np.ndarray, np.ndarray]:
    acts = np.load(bank / "activations.npy", mmap_mode="r")
    hue: list[float] = []
    with (bank / "prompts.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            r, g, b = row["rgb"]
            h, _s, _v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            hue.append(TAU * h)
    hue_arr = np.asarray(hue, dtype=np.float64)
    if acts.shape[0] != hue_arr.shape[0]:
        raise ValueError(
            f"bank mismatch: {acts.shape[0]} activation rows vs "
            f"{hue_arr.shape[0]} prompt rows"
        )
    if acts.ndim != 3:
        raise ValueError(f"activations.npy must be 3D, got shape {acts.shape}")
    return acts, hue_arr


def parse_layers(raw: str | None, n_layers: int) -> list[int]:
    if raw is None or raw.strip() == "":
        layers = DEFAULT_LAYERS
    else:
        layers = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(layers) < 3:
        raise ValueError("--layers must name at least three layers")
    if len(set(layers)) != len(layers):
        raise ValueError(f"--layers contains duplicates: {layers}")
    for layer in layers:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"layer {layer} outside activation range [0, {n_layers})")
    return layers


def deterministic_split(
    n_rows: int,
    eval_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    n_eval = max(1, int(round(n_rows * eval_fraction)))
    n_eval = min(n_eval, n_rows - 1)
    return np.sort(order[n_eval:]), np.sort(order[:n_eval])


def periodic_design(theta: np.ndarray, n_harmonics: int) -> np.ndarray:
    cols = [np.ones(theta.shape[0], dtype=np.float64)]
    for harmonic in range(1, n_harmonics + 1):
        angle = harmonic * theta
        cols.append(np.cos(angle))
        cols.append(np.sin(angle))
    return np.column_stack(cols)


def explained_variance(x: np.ndarray, fitted: np.ndarray) -> float:
    residual = float(np.square(x - fitted).sum())
    centered = x - x.mean(axis=0, keepdims=True)
    total = float(np.square(centered).sum())
    return 1.0 - residual / total if total > 0.0 else 0.0


def align_phase_to_hue(theta: np.ndarray, hue: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    best: tuple[float, int, float, np.ndarray] | None = None
    for sign in (-1, 1):
        shifted = sign * theta
        offset = math.atan2(np.sin(hue - shifted).mean(), np.cos(hue - shifted).mean())
        aligned = np.mod(shifted + offset, TAU)
        corr = circular_corr(aligned, hue)
        score = abs(corr)
        if best is None or score > best[0]:
            best = (score, sign, offset, aligned)
    assert best is not None
    score, sign, offset, aligned = best
    return aligned, {
        "gauge_sign": sign,
        "gauge_offset_rad": json_float(offset),
        "gauge_abs_hue_circular_corr": json_float(score),
    }


def standardize_layer(acts: np.ndarray, layer: int) -> np.ndarray:
    x = np.asarray(acts[:, layer, :], dtype=np.float64)
    mu = x.mean(axis=0, keepdims=True)
    sd = np.maximum(x.std(axis=0, keepdims=True), 1e-6)
    return np.ascontiguousarray((x - mu) / sd)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)


def predict_vector(model: Any, theta: np.ndarray) -> np.ndarray:
    pred = model.predict({"theta": np.asarray(theta, dtype=np.float64)}, return_type="dict")
    if isinstance(pred, dict):
        if "mean" in pred:
            return np.asarray(pred["mean"], dtype=np.float64)
        if "linear_predictor" in pred:
            return np.asarray(pred["linear_predictor"], dtype=np.float64)
    return np.asarray(pred, dtype=np.float64)


def predict_map(fit: TransportFit, theta: np.ndarray) -> np.ndarray:
    pred_cos = predict_vector(fit.cos_model, theta)
    pred_sin = predict_vector(fit.sin_model, theta)
    return np.mod(np.arctan2(pred_sin, pred_cos), TAU)


def angle_error_summary(prefix: str, pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    err = circular_abs_error(pred, truth)
    corr = circular_corr(pred, truth) if err.shape[0] >= 3 else 0.0
    return {
        f"{prefix}_n": int(err.shape[0]),
        f"{prefix}_circular_corr": json_float(corr),
        f"{prefix}_mean_abs_error_rad": json_float(err.mean()),
        f"{prefix}_median_abs_error_rad": json_float(np.median(err)),
        f"{prefix}_rms_error_rad": json_float(math.sqrt(float(np.mean(err * err)))),
        f"{prefix}_max_abs_error_rad": json_float(err.max()),
    }


def map_grid_metrics(
    cos_model: Any,
    sin_model: Any,
    grid_size: int,
) -> dict[str, Any]:
    grid = np.linspace(0.0, TAU, grid_size, endpoint=False)
    step = TAU / float(grid_size)
    map_grid = np.mod(
        np.arctan2(predict_vector(sin_model, grid), predict_vector(cos_model, grid)),
        TAU,
    )
    map_plus = np.mod(
        np.arctan2(
            predict_vector(sin_model, np.mod(grid + step, TAU)),
            predict_vector(cos_model, np.mod(grid + step, TAU)),
        ),
        TAU,
    )
    map_minus = np.mod(
        np.arctan2(
            predict_vector(sin_model, np.mod(grid - step, TAU)),
            predict_vector(cos_model, np.mod(grid - step, TAU)),
        ),
        TAU,
    )
    deriv = angle_delta(map_plus, map_minus) / (2.0 * step)
    curvature = (np.roll(deriv, -1) - np.roll(deriv, 1)) / (2.0 * step)
    defect = np.abs(np.abs(deriv) - 1.0)

    closed = np.concatenate([map_grid, map_grid[:1]])
    winding = float(angle_delta(closed[1:], closed[:-1]).sum() / TAU)
    winding_round = int(np.rint(winding))
    if winding_round > 0:
        winding_sign = 1
    elif winding_round < 0:
        winding_sign = -1
    else:
        winding_sign = 0

    abs_curvature = np.abs(curvature)
    return {
        "isometry_defect_mean": json_float(defect.mean()),
        "isometry_defect_rms": json_float(math.sqrt(float(np.mean(defect * defect)))),
        "isometry_defect_max": json_float(defect.max()),
        "derivative_abs_mean": json_float(np.abs(deriv).mean()),
        "derivative_mean": json_float(deriv.mean()),
        "curvature_abs_mean": json_float(abs_curvature.mean()),
        "curvature_abs_rms": json_float(math.sqrt(float(np.mean(curvature * curvature)))),
        "curvature_abs_max": json_float(abs_curvature.max()),
        "winding_number": winding,
        "winding_number_round": winding_round,
        "winding_number_sign": winding_sign,
    }


def fit_layer_chart(
    bank: str,
    hue: np.ndarray,
    layer: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    t0 = time.time()
    acts = np.load(Path(bank) / "activations.npy", mmap_mode="r")
    x = standardize_layer(acts, layer)
    centered = x - x.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    if u.shape[1] < 2:
        raise ValueError(f"layer {layer} PCA chart needs at least two components")
    raw_theta = np.mod(np.arctan2(u[:, 1] * s[1], u[:, 0] * s[0]), TAU)
    theta, gauge_info = align_phase_to_hue(raw_theta, hue)
    design = periodic_design(theta, n_harmonics=3)
    decoder, *_ = np.linalg.lstsq(design, x, rcond=None)
    fitted = design @ decoder
    corr = circular_corr(theta, hue)
    entry = {
        "layer": layer,
        "seconds": time.time() - t0,
        "n": int(x.shape[0]),
        "p": int(x.shape[1]),
        "chart_method": "pca_periodic_atom_hue_gauge",
        "decoder_basis": "periodic",
        "decoder_harmonics": 3,
        "hue_circular_corr": corr,
        "abs_hue_circular_corr": abs(corr),
        "reml_score": None,
        "reconstruction_r2": json_float(explained_variance(x, fitted)),
        "dispersion": json_float(np.square(x - fitted).mean()),
        "active_dim": 1,
        "pca_singular_values": [json_float(v) for v in s[:8]],
        "coordinate_units": "radians",
        "raw_min": json_float(raw_theta.min()),
        "raw_max": json_float(raw_theta.max()),
        "raw_mean": json_float(raw_theta.mean()),
        **gauge_info,
    }
    return theta, entry


def fit_transport(
    theta_current: np.ndarray,
    theta_next: np.ndarray,
    layer_current: int,
    layer_next: int,
    basis_dim: int,
    grid_size: int,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    map_kind: str,
) -> TransportFit:
    t0 = time.time()
    import gamfit

    data = {
        "theta": theta_current[train_idx],
        "cos_next": np.cos(theta_next[train_idx]),
        "sin_next": np.sin(theta_next[train_idx]),
    }
    smooth = f"s(theta, periodic=true, k={basis_dim}, period={TAU}, origin=0)"
    cos_model = gamfit.fit(data, f"cos_next ~ {smooth}", family="gaussian")
    sin_model = gamfit.fit(data, f"sin_next ~ {smooth}", family="gaussian")

    fit = TransportFit(
        layer_current=layer_current,
        layer_next=layer_next,
        cos_model=cos_model,
        sin_model=sin_model,
        entry={},
    )
    train_mapped = predict_map(fit, theta_current[train_idx])
    eval_mapped = predict_map(fit, theta_current[eval_idx])
    cos_summary = cos_model.summary()
    sin_summary = sin_model.summary()
    eval_cos = predict_vector(cos_model, theta_current[eval_idx])
    eval_sin = predict_vector(sin_model, theta_current[eval_idx])
    radius = np.sqrt(eval_cos * eval_cos + eval_sin * eval_sin)
    entry = {
        "layer_pair": [layer_current, layer_next],
        "map_kind": map_kind,
        "pair_stride": layer_next - layer_current,
        "seconds": time.time() - t0,
        "basis_dim": basis_dim,
        "grid_size": grid_size,
        **angle_error_summary("train", train_mapped, theta_next[train_idx]),
        **angle_error_summary("heldout", eval_mapped, theta_next[eval_idx]),
        **map_grid_metrics(cos_model, sin_model, grid_size),
        "predicted_radius_mean": json_float(radius.mean()),
        "predicted_radius_min": json_float(radius.min()),
        "cos_reml_score": json_float(cos_summary["reml_score"]),
        "sin_reml_score": json_float(sin_summary["reml_score"]),
    }
    fit.entry = entry
    return fit


def composition_entry(
    left: TransportFit,
    middle: TransportFit,
    direct: TransportFit,
    theta_source: np.ndarray,
    theta_target: np.ndarray,
    eval_idx: np.ndarray,
    grid_size: int,
) -> dict[str, Any]:
    t0 = time.time()
    heldout_source = theta_source[eval_idx]
    heldout_target = theta_target[eval_idx]
    direct_heldout = predict_map(direct, heldout_source)
    composed_heldout = predict_map(middle, predict_map(left, heldout_source))
    residual = circular_abs_error(direct_heldout, composed_heldout)

    grid = np.linspace(0.0, TAU, grid_size, endpoint=False)
    direct_grid = predict_map(direct, grid)
    composed_grid = predict_map(middle, predict_map(left, grid))
    grid_residual = circular_abs_error(direct_grid, composed_grid)

    return {
        "layer_triple": [
            left.layer_current,
            left.layer_next,
            middle.layer_next,
        ],
        "direct_layer_pair": [
            direct.layer_current,
            direct.layer_next,
        ],
        "seconds": time.time() - t0,
        **angle_error_summary(
            "direct_heldout",
            direct_heldout,
            heldout_target,
        ),
        **angle_error_summary(
            "composed_heldout",
            composed_heldout,
            heldout_target,
        ),
        "heldout_direct_composed_circular_corr": json_float(
            circular_corr(direct_heldout, composed_heldout)
        ),
        "heldout_composition_residual_mean_rad": json_float(residual.mean()),
        "heldout_composition_residual_median_rad": json_float(np.median(residual)),
        "heldout_composition_residual_rms_rad": json_float(
            math.sqrt(float(np.mean(residual * residual)))
        ),
        "heldout_composition_residual_max_rad": json_float(residual.max()),
        "grid_composition_residual_mean_rad": json_float(grid_residual.mean()),
        "grid_composition_residual_rms_rad": json_float(
            math.sqrt(float(np.mean(grid_residual * grid_residual)))
        ),
        "grid_composition_residual_max_rad": json_float(grid_residual.max()),
    }


def write_profile(
    prefix: Path,
    transports: list[dict[str, Any]],
    compositions: list[dict[str, Any]],
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    adjacent = [
        item
        for item in transports
        if item.get("status") == "ok" and item.get("map_kind") == "adjacent"
    ]
    adjacent.sort(key=lambda item: item["layer_pair"])
    rows = [
        {
            "from_layer": item["layer_pair"][0],
            "to_layer": item["layer_pair"][1],
            "heldout_mean_abs_error_rad": item["heldout_mean_abs_error_rad"],
            "isometry_defect_mean": item["isometry_defect_mean"],
            "isometry_defect_rms": item["isometry_defect_rms"],
            "isometry_defect_max": item["isometry_defect_max"],
            "curvature_abs_mean": item["curvature_abs_mean"],
            "curvature_abs_rms": item["curvature_abs_rms"],
            "curvature_abs_max": item["curvature_abs_max"],
            "winding_number": item["winding_number"],
            "heldout_circular_corr": item["heldout_circular_corr"],
        }
        for item in adjacent
    ]
    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    comp_rows = [
        {
            "from_layer": item["layer_triple"][0],
            "middle_layer": item["layer_triple"][1],
            "to_layer": item["layer_triple"][2],
            "heldout_composition_residual_mean_rad": item[
                "heldout_composition_residual_mean_rad"
            ],
            "heldout_composition_residual_rms_rad": item[
                "heldout_composition_residual_rms_rad"
            ],
            "grid_composition_residual_mean_rad": item[
                "grid_composition_residual_mean_rad"
            ],
        }
        for item in compositions
        if item.get("status") == "ok"
    ]
    comp_csv_path = prefix.with_name(prefix.name + "_composition").with_suffix(".csv")
    with comp_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comp_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [0.5 * (row["from_layer"] + row["to_layer"]) for row in rows]
    x_comp = [row["middle_layer"] for row in comp_rows]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=False)
    axes[0].plot(
        x,
        [row["curvature_abs_mean"] for row in rows],
        marker="o",
        label="mean |d2h/dtheta2|",
    )
    axes[0].plot(
        x,
        [row["isometry_defect_mean"] for row in rows],
        marker="s",
        label="mean ||dh/dtheta|-1|",
    )
    axes[0].set_xlabel("adjacent layer-pair midpoint")
    axes[0].set_ylabel("grid diagnostic")
    axes[0].set_title("Adjacent transport curvature and isometry defect")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        x_comp,
        [row["heldout_composition_residual_mean_rad"] for row in comp_rows],
        marker="o",
        label="held-out mean",
    )
    axes[1].plot(
        x_comp,
        [row["heldout_composition_residual_rms_rad"] for row in comp_rows],
        marker="s",
        label="held-out rms",
    )
    axes[1].set_xlabel("middle layer in l -> l+1 -> l+2")
    axes[1].set_ylabel("composition residual (rad)")
    axes[1].set_title("Functoriality residual: direct vs composed two-hop map")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default=DEFAULT_BANK)
    parser.add_argument("--layers", default=None, help="comma-separated layer ladder")
    parser.add_argument("--seed", type=int, default=1013)
    parser.add_argument("--transport-k", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--out", default=DEFAULT_REPORT)
    parser.add_argument("--profile-prefix", default=DEFAULT_PROFILE_PREFIX)
    args = parser.parse_args()

    bank = Path(args.bank)
    out = Path(args.out)
    profile_prefix = Path(args.profile_prefix)
    acts, hue = load_bank(bank)
    layers = parse_layers(args.layers, int(acts.shape[1]))
    train_idx, eval_idx = deterministic_split(
        int(acts.shape[0]),
        DEFAULT_EVAL_FRACTION,
        args.seed,
    )
    report: dict[str, Any] = {
        "bank": str(bank),
        "activation_shape": [int(v) for v in acts.shape],
        "layers": layers,
        "seed": args.seed,
        "transport_basis_dim": args.transport_k,
        "grid_size": args.grid_size,
        "eval_fraction": DEFAULT_EVAL_FRACTION,
        "train_n": int(train_idx.shape[0]),
        "heldout_n": int(eval_idx.shape[0]),
        "layer_fits": [],
        "transports": [],
        "composition_tests": [],
        "started_at_unix": time.time(),
    }
    print(
        f"[setup] bank={bank} acts={acts.shape} layers={layers} "
        f"chart=pca_periodic_atom_hue_gauge",
        flush=True,
    )
    write_report(out, report)

    theta_by_layer: dict[int, np.ndarray] = {}
    for layer in layers:
        print(f"[chart] fitting layer {layer}", flush=True)
        try:
            theta, entry = fit_layer_chart(
                str(bank),
                hue,
                layer,
            )
        except Exception as exc:
            entry = {
                "layer": layer,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
            }
            report["layer_fits"].append(entry)
            write_report(out, report)
            print(f"[chart] layer {layer} status=error", flush=True)
            raise
        entry["status"] = "ok"
        theta_by_layer[layer] = theta
        report["layer_fits"].append(entry)
        report["layer_fits"].sort(key=lambda item: item["layer"])
        write_report(out, report)
        print(
            f"[chart] layer {layer} corr={entry['hue_circular_corr']:.4f} "
            f"r2={entry['reconstruction_r2']} {entry['seconds']:.1f}s",
            flush=True,
        )

    adjacent_pairs = list(zip(layers[:-1], layers[1:]))
    twohop_pairs = list(zip(layers[:-2], layers[2:]))
    transport_pairs = [
        ("adjacent", left, right)
        for left, right in adjacent_pairs
    ] + [
        ("direct_twohop", left, right)
        for left, right in twohop_pairs
    ]
    fits: dict[tuple[int, int], TransportFit] = {}
    for map_kind, left, right in transport_pairs:
        print(f"[transport] fitting {left}->{right} kind={map_kind}", flush=True)
        try:
            fit = fit_transport(
                theta_by_layer[left],
                theta_by_layer[right],
                left,
                right,
                args.transport_k,
                args.grid_size,
                train_idx,
                eval_idx,
                map_kind,
            )
        except Exception as exc:
            entry = {
                "layer_pair": [left, right],
                "map_kind": map_kind,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
            }
            report["transports"].append(entry)
            write_report(out, report)
            print(f"[transport] {left}->{right} status=error", flush=True)
            raise
        entry = fit.entry
        entry["status"] = "ok"
        fits[(left, right)] = fit
        report["transports"].append(entry)
        report["transports"].sort(
            key=lambda item: (item["layer_pair"][0], item["layer_pair"][1])
        )
        write_report(out, report)
        print(
            f"[transport] {left}->{right} heldout_mae="
            f"{entry['heldout_mean_abs_error_rad']:.4f} curvature="
            f"{entry['curvature_abs_mean']:.4f} winding="
            f"{entry['winding_number']:.3f} {entry['seconds']:.1f}s",
            flush=True,
        )

    for left, middle, right in zip(layers[:-2], layers[1:-1], layers[2:]):
        print(f"[composition] testing {left}->{middle}->{right}", flush=True)
        try:
            entry = composition_entry(
                fits[(left, middle)],
                fits[(middle, right)],
                fits[(left, right)],
                theta_by_layer[left],
                theta_by_layer[right],
                eval_idx,
                args.grid_size,
            )
        except Exception as exc:
            entry = {
                "layer_triple": [left, middle, right],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
            }
            report["composition_tests"].append(entry)
            write_report(out, report)
            print(f"[composition] {left}->{middle}->{right} status=error", flush=True)
            raise
        entry["status"] = "ok"
        report["composition_tests"].append(entry)
        write_report(out, report)
        print(
            f"[composition] {left}->{middle}->{right} residual_mean="
            f"{entry['heldout_composition_residual_mean_rad']:.4f} "
            f"residual_rms={entry['heldout_composition_residual_rms_rad']:.4f}",
            flush=True,
        )

    write_profile(profile_prefix, report["transports"], report["composition_tests"])
    report["profile_csv"] = str(profile_prefix.with_suffix(".csv"))
    report["composition_csv"] = str(
        profile_prefix.with_name(profile_prefix.name + "_composition").with_suffix(".csv")
    )
    report["profile_png"] = str(profile_prefix.with_suffix(".png"))
    report["finished_at_unix"] = time.time()
    report["seconds"] = report["finished_at_unix"] - report["started_at_unix"]
    write_report(out, report)
    print(f"[done] report={out}", flush=True)
    print(f"[done] profile={profile_prefix.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
