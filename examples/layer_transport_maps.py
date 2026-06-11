#!/usr/bin/env python3
"""Layer-resolved circle-coordinate transport on the OLMo color bank.

This experiment fits a K=1 circle chart to the last-token activations at a
small ladder of transformer layers, then estimates smooth circular transport
maps between adjacent ladder layers:

    theta_next = h(theta_current)

The transport map is represented by two periodic GAMs for cos(theta_next) and
sin(theta_next). The mapped angle is recovered with atan2, which keeps the
response circular while using the scalar Gaussian GAM API.
"""

from __future__ import annotations

import argparse
import colorsys
import concurrent.futures
import csv
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


TAU = 2.0 * math.pi
DEFAULT_BANK = (
    "/home/azuser/Manifold-SAE/runs/OLMO3_32B_TRAJ_SFT/"
    "5e-5-step10790/extra"
)
DEFAULT_REPORT = "/mnt/work/exp/layer_transport_report.json"
DEFAULT_PROFILE_PREFIX = "/mnt/work/exp/layer_transport_isometry_profile"


def json_float(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def angle_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (a - b)))


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
        layers = list(range(4, n_layers, 8))
        if len(layers) > 8:
            layers = layers[:8]
        return layers
    layers = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(layers) < 2:
        raise ValueError("--layers must name at least two layers")
    if len(set(layers)) != len(layers):
        raise ValueError(f"--layers contains duplicates: {layers}")
    for layer in layers:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"layer {layer} outside activation range [0, {n_layers})")
    return layers


def standardize_layer(acts: np.ndarray, layer: int) -> np.ndarray:
    x = np.asarray(acts[:, layer, :], dtype=np.float64)
    mu = x.mean(axis=0, keepdims=True)
    sd = np.maximum(x.std(axis=0, keepdims=True), 1e-6)
    return np.ascontiguousarray((x - mu) / sd)


def phase_to_radians(coords: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(coords, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 1:
        raise ValueError(f"K=1 circle coordinates must have shape (N, 1), got {raw.shape}")
    phase = raw[:, 0]
    info = {
        "raw_min": json_float(phase.min()),
        "raw_max": json_float(phase.max()),
        "raw_mean": json_float(phase.mean()),
    }
    if float(phase.min()) >= -0.25 and float(phase.max()) <= 1.25:
        info["coordinate_units"] = "normalized_phase"
        theta = np.mod(phase, 1.0) * TAU
    else:
        info["coordinate_units"] = "radians"
        theta = np.mod(phase, TAU)
    return theta.astype(np.float64, copy=False), info


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


def worker_gamfit() -> Any:
    os.environ["RAYON_NUM_THREADS"] = "4"
    import gamfit

    return gamfit


def fit_layer_chart(
    bank: str,
    hue: np.ndarray,
    layer: int,
    n_iter: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    t0 = time.time()
    gamfit = worker_gamfit()
    acts = np.load(Path(bank) / "activations.npy", mmap_mode="r")
    x = standardize_layer(acts, layer)
    model = gamfit.sae_manifold_fit(
        X=x,
        K=1,
        d_atom=1,
        atom_topology="circle",
        n_iter=n_iter,
        random_state=seed,
    )
    theta, coord_info = phase_to_radians(np.asarray(model.coords[0], dtype=np.float64))
    corr = circular_corr(theta, hue)
    entry = {
        "layer": layer,
        "seconds": time.time() - t0,
        "n": int(x.shape[0]),
        "p": int(x.shape[1]),
        "hue_circular_corr": corr,
        "abs_hue_circular_corr": abs(corr),
        "reml_score": json_float(getattr(model, "reml_score", None)),
        "reconstruction_r2": json_float(getattr(model, "reconstruction_r2", None)),
        "dispersion": json_float(getattr(model, "dispersion", None)),
        "active_dim": getattr(model.atoms[0], "active_dim", None),
        **coord_info,
    }
    return theta, entry


def fit_transport(
    theta_current: np.ndarray,
    theta_next: np.ndarray,
    layer_current: int,
    layer_next: int,
    basis_dim: int,
    grid_size: int,
) -> dict[str, Any]:
    t0 = time.time()
    gamfit = worker_gamfit()
    data = {
        "theta": theta_current,
        "cos_next": np.cos(theta_next),
        "sin_next": np.sin(theta_next),
    }
    smooth = f"s(theta, periodic=true, k={basis_dim}, period={TAU}, origin=0)"
    cos_model = gamfit.fit(data, f"cos_next ~ {smooth}", family="gaussian")
    sin_model = gamfit.fit(data, f"sin_next ~ {smooth}", family="gaussian")

    pred_cos = predict_vector(cos_model, theta_current)
    pred_sin = predict_vector(sin_model, theta_current)
    mapped = np.mod(np.arctan2(pred_sin, pred_cos), TAU)
    corr = circular_corr(mapped, theta_next)

    grid = np.linspace(0.0, TAU, grid_size, endpoint=False)
    eps = 1e-3
    plus = np.mod(grid + eps, TAU)
    minus = np.mod(grid - eps, TAU)
    map_grid = np.mod(
        np.arctan2(predict_vector(sin_model, grid), predict_vector(cos_model, grid)),
        TAU,
    )
    map_plus = np.mod(
        np.arctan2(predict_vector(sin_model, plus), predict_vector(cos_model, plus)),
        TAU,
    )
    map_minus = np.mod(
        np.arctan2(predict_vector(sin_model, minus), predict_vector(cos_model, minus)),
        TAU,
    )
    deriv = angle_delta(map_plus, map_minus) / (2.0 * eps)
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

    cos_summary = cos_model.summary()
    sin_summary = sin_model.summary()
    radius = np.sqrt(pred_cos * pred_cos + pred_sin * pred_sin)
    return {
        "layer_pair": [layer_current, layer_next],
        "seconds": time.time() - t0,
        "basis_dim": basis_dim,
        "grid_size": grid_size,
        "mapped_actual_circular_corr": corr,
        "abs_mapped_actual_circular_corr": abs(corr),
        "isometry_defect_mean": json_float(defect.mean()),
        "isometry_defect_rms": json_float(math.sqrt(float(np.mean(defect * defect)))),
        "isometry_defect_max": json_float(defect.max()),
        "derivative_abs_mean": json_float(np.abs(deriv).mean()),
        "derivative_mean": json_float(deriv.mean()),
        "winding_number": winding,
        "winding_number_round": winding_round,
        "winding_number_sign": winding_sign,
        "predicted_radius_mean": json_float(radius.mean()),
        "predicted_radius_min": json_float(radius.min()),
        "cos_reml_score": json_float(cos_summary["reml_score"]),
        "sin_reml_score": json_float(sin_summary["reml_score"]),
    }


def write_profile(prefix: Path, transports: list[dict[str, Any]]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "from_layer": item["layer_pair"][0],
            "to_layer": item["layer_pair"][1],
            "isometry_defect_mean": item["isometry_defect_mean"],
            "isometry_defect_rms": item["isometry_defect_rms"],
            "isometry_defect_max": item["isometry_defect_max"],
            "winding_number": item["winding_number"],
            "mapped_actual_circular_corr": item["mapped_actual_circular_corr"],
        }
        for item in transports
    ]
    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [0.5 * (row["from_layer"] + row["to_layer"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, [row["isometry_defect_mean"] for row in rows], marker="o", label="mean")
    ax.plot(x, [row["isometry_defect_rms"] for row in rows], marker="s", label="rms")
    ax.set_xlabel("layer-pair midpoint")
    ax.set_ylabel("isometry defect | |dh/dtheta| - 1 |")
    ax.set_title("Layer transport isometry defect")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default=DEFAULT_BANK)
    parser.add_argument("--layers", default=None, help="comma-separated layer ladder")
    parser.add_argument("--n-iter", type=int, default=24)
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
    report: dict[str, Any] = {
        "bank": str(bank),
        "activation_shape": [int(v) for v in acts.shape],
        "layers": layers,
        "n_iter": args.n_iter,
        "seed": args.seed,
        "transport_basis_dim": args.transport_k,
        "grid_size": args.grid_size,
        "layer_fits": [],
        "transports": [],
        "started_at_unix": time.time(),
    }
    print(
        f"[setup] bank={bank} acts={acts.shape} layers={layers} "
        f"n_iter={args.n_iter}",
        flush=True,
    )
    write_report(out, report)

    theta_by_layer: dict[int, np.ndarray] = {}
    chart_failed = False
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(len(layers), 4)
    ) as executor:
        chart_futures = {}
        for layer in layers:
            print(f"[chart] fitting layer {layer}", flush=True)
            future = executor.submit(
                fit_layer_chart,
                str(bank),
                hue,
                layer,
                args.n_iter,
                args.seed + layer,
            )
            chart_futures[future] = layer

        for future in concurrent.futures.as_completed(chart_futures):
            layer = chart_futures[future]
            try:
                theta, entry = future.result()
            except Exception as exc:
                chart_failed = True
                entry = {
                    "layer": layer,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-4000:],
                }
            else:
                entry["status"] = "ok"
                theta_by_layer[layer] = theta
            report["layer_fits"].append(entry)
            write_report(out, report)
            if entry["status"] == "ok":
                print(
                    f"[chart] layer {layer} corr={entry['hue_circular_corr']:.4f} "
                    f"r2={entry['reconstruction_r2']} {entry['seconds']:.1f}s",
                    flush=True,
                )
            else:
                print(f"[chart] layer {layer} status=error", flush=True)
    if chart_failed:
        raise RuntimeError("one or more layer chart fits failed")

    pairs = list(zip(layers[:-1], layers[1:]))
    transport_failed = False
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(len(pairs), 4)
    ) as executor:
        transport_futures = {}
        for left, right in pairs:
            print(f"[transport] fitting {left}->{right}", flush=True)
            future = executor.submit(
                fit_transport,
                theta_by_layer[left],
                theta_by_layer[right],
                left,
                right,
                args.transport_k,
                args.grid_size,
            )
            transport_futures[future] = (left, right)

        for future in concurrent.futures.as_completed(transport_futures):
            left, right = transport_futures[future]
            try:
                entry = future.result()
            except Exception as exc:
                transport_failed = True
                entry = {
                    "layer_pair": [left, right],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-4000:],
                }
            else:
                entry["status"] = "ok"
            report["transports"].append(entry)
            write_report(out, report)
            if entry["status"] == "ok":
                print(
                    f"[transport] {left}->{right} "
                    f"corr={entry['mapped_actual_circular_corr']:.4f} "
                    f"defect_mean={entry['isometry_defect_mean']:.4f} "
                    f"winding={entry['winding_number']:.3f} {entry['seconds']:.1f}s",
                    flush=True,
                )
            else:
                print(f"[transport] {left}->{right} status=error", flush=True)
    if transport_failed:
        raise RuntimeError("one or more transport fits failed")

    write_profile(profile_prefix, report["transports"])
    report["profile_csv"] = str(profile_prefix.with_suffix(".csv"))
    report["profile_png"] = str(profile_prefix.with_suffix(".png"))
    report["finished_at_unix"] = time.time()
    report["seconds"] = report["finished_at_unix"] - report["started_at_unix"]
    write_report(out, report)
    print(f"[done] report={out}", flush=True)
    print(f"[done] profile={profile_prefix.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
