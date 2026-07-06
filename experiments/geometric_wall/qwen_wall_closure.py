#!/usr/bin/env python3
"""Geometric-Wall closure test on existing Qwen residual activations.

This script is intentionally numpy-only. It samples rows from pre-harvested
activation arrays, peels the leading PCA sink direction, then compares
matched-parameter flat and quadratic local reconstructions.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np


class LayerSpec(NamedTuple):
    label: str
    path: Path


class PcaSketch(NamedTuple):
    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    total_energy: float


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_layer_specs(raw_specs: List[str]) -> List[LayerSpec]:
    specs = []  # type: List[LayerSpec]
    for raw in raw_specs:
        if ":" not in raw:
            raise SystemExit(f"layer spec must be LABEL:PATH, got {raw!r}")
        label, path = raw.split(":", 1)
        label = label.strip()
        if not label:
            raise SystemExit(f"layer label is empty in {raw!r}")
        specs.append(LayerSpec(label=label, path=Path(path)))
    if not specs:
        raise SystemExit("at least one --layer is required")
    return specs


def sample_rows(path: Path, n_rows: int, seed: int) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise SystemExit(f"{path} must be a rank-2 array, got shape {arr.shape}")
    take = min(n_rows, int(arr.shape[0]))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(arr.shape[0], size=take, replace=False))
    return np.asarray(arr[indices], dtype=np.float32)


def randomized_pca(
    x: np.ndarray,
    n_components: int,
    oversample: int,
    power_iter: int,
    seed: int,
) -> PcaSketch:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    xc = x.astype(np.float32, copy=True)
    xc -= mean
    total_energy = float(np.sum(xc.astype(np.float64) * xc.astype(np.float64)))
    rank = min(n_components + oversample, xc.shape[0] - 1, xc.shape[1])
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((xc.shape[1], rank)).astype(np.float32)
    y = xc @ omega
    for _iteration in range(power_iter):
        z = xc.T @ y
        y = xc @ z
    q, _r = np.linalg.qr(y, mode="reduced")
    b = q.T @ xc
    _u, s, vt = np.linalg.svd(b, full_matrices=False)
    keep = min(n_components, vt.shape[0])
    return PcaSketch(
        mean=mean,
        components=vt[:keep].astype(np.float32),
        singular_values=s[:keep].astype(np.float64),
        total_energy=total_energy,
    )


def peel_sink(x: np.ndarray, sketch: PcaSketch) -> Tuple[np.ndarray, float]:
    centered = x.astype(np.float32, copy=True)
    centered -= sketch.mean
    sink = sketch.components[0]
    sink_scores = centered @ sink
    centered -= sink_scores[:, None] * sink[None, :]
    sink_fraction = float((sketch.singular_values[0] ** 2) / max(sketch.total_energy, 1e-30))
    return centered, sink_fraction


def quadratic_features(z: np.ndarray) -> np.ndarray:
    q = z.shape[1]
    cols = [z[:, i] * z[:, j] for i in range(q) for j in range(i, q)]
    return np.stack(cols, axis=1).astype(np.float32)


def ridge_coefficients(design: np.ndarray, target: np.ndarray, ridge_scale: float) -> np.ndarray:
    x64 = design.astype(np.float64)
    y64 = target.astype(np.float64)
    gram = x64.T @ x64
    scale = float(np.trace(gram) / max(gram.shape[0], 1))
    ridge = ridge_scale * max(scale, 1e-30)
    rhs = x64.T @ y64
    gram.flat[:: gram.shape[0] + 1] += ridge
    return np.linalg.solve(gram, rhs).astype(np.float32)


def pca_basis(x: np.ndarray, rank: int) -> np.ndarray:
    x32 = x.astype(np.float32)
    gram = x32 @ x32.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    keep = min(rank, int(np.count_nonzero(values > max(values[0], 0.0) * 1e-12)))
    if keep == 0:
        return np.zeros((0, x.shape[1]), dtype=np.float32)
    scaled = vectors[:, :keep].T @ x32
    scaled /= np.sqrt(values[:keep])[:, None]
    return scaled.astype(np.float32)


def neighborhood_floors(
    train: np.ndarray,
    test: np.ndarray,
    tangent_rank: int,
    flat_rank: int,
    ridge_scale: float,
) -> Tuple[float, float, float]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    train_centered = train.astype(np.float32) - mean
    test_centered = test.astype(np.float32) - mean
    basis = pca_basis(train_centered, flat_rank)
    flat_scores = test_centered @ basis.T
    flat_recon = flat_scores @ basis
    flat = energy_ratio(test_centered - flat_recon, test_centered)
    tangent = basis[:tangent_rank]
    z_train = train_centered @ tangent.T
    z_test = test_centered @ tangent.T
    train_linear = z_train @ tangent
    test_linear = z_test @ tangent
    train_resid = train_centered - train_linear
    phi_train = quadratic_features(z_train)
    phi_mean = phi_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    phi_train -= phi_mean
    beta = ridge_coefficients(phi_train, train_resid, ridge_scale)
    phi_test = quadratic_features(z_test)
    phi_test -= phi_mean
    correction = phi_test @ beta
    resid = test_centered - test_linear - correction
    floor = energy_ratio(resid, test_centered)
    curvature = math.sqrt(
        float(np.sum(correction.astype(np.float64) * correction.astype(np.float64)))
        / max(float(np.sum(test_centered.astype(np.float64) * test_centered.astype(np.float64))), 1e-30)
    )
    return flat, floor, curvature


def energy_ratio(resid: np.ndarray, target: np.ndarray) -> float:
    num = float(np.sum(resid.astype(np.float64) * resid.astype(np.float64)))
    den = float(np.sum(target.astype(np.float64) * target.astype(np.float64)))
    return num / max(den, 1e-30)


def standardized_scores(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    scores = x @ basis.T
    scale = scores.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-8)
    return (scores / scale).astype(np.float32)


def neighborhood_indices(
    scores: np.ndarray,
    n_neighborhoods: int,
    neighborhood_size: int,
    seed: int,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    anchors = rng.choice(scores.shape[0], size=min(n_neighborhoods, scores.shape[0]), replace=False)
    neighborhoods = []  # type: List[np.ndarray]
    for anchor in anchors:
        delta = scores - scores[int(anchor)]
        distances = np.sum(delta.astype(np.float64) * delta.astype(np.float64), axis=1)
        take = min(neighborhood_size, scores.shape[0])
        idx = np.argpartition(distances, take - 1)[:take]
        neighborhoods.append(idx.astype(np.int64))
    return neighborhoods


def split_neighborhood(idx: np.ndarray, train_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = idx.copy()
    rng.shuffle(perm)
    split = int(round(train_fraction * perm.shape[0]))
    split = min(max(split, 2), perm.shape[0] - 1)
    return perm[:split], perm[split:]


def correlation(xs: List[float], ys: List[float]) -> Optional[float]:
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


def layer_experiment(args: argparse.Namespace, spec: LayerSpec, layer_index: int) -> Dict[str, Any]:
    log(f"{spec.label}: loading deterministic sample")
    x = sample_rows(spec.path, args.sample_rows, args.seed + 1009 * layer_index)
    q = args.tangent_rank
    quad_cols = q * (q + 1) // 2
    flat_rank = q + quad_cols
    needed_components = max(flat_rank + 1, args.neighbor_dims + 1)
    log(f"{spec.label}: randomized PCA for sink and neighborhood coordinates")
    sketch = randomized_pca(
        x,
        n_components=needed_components,
        oversample=args.pca_oversample,
        power_iter=args.power_iter,
        seed=args.seed + 7919 * (layer_index + 1),
    )
    peeled, sink_fraction = peel_sink(x, sketch)
    basis = sketch.components[1 : args.neighbor_dims + 1]
    scores = standardized_scores(peeled, basis)
    neighborhoods = neighborhood_indices(
        scores,
        args.neighborhoods,
        args.neighborhood_size,
        args.seed + 3571 * (layer_index + 1),
    )
    rows = []  # type: List[Dict[str, float]]
    log(f"{spec.label}: fitting {len(neighborhoods)} local flat/curved closures")
    progress_stride = max(1, len(neighborhoods) // 6)
    for i, idx in enumerate(neighborhoods):
        train_idx, test_idx = split_neighborhood(
            idx,
            args.train_fraction,
            args.seed + 104729 * (layer_index + 1) + i,
        )
        train = peeled[train_idx]
        test = peeled[test_idx]
        flat, curved, curv = neighborhood_floors(train, test, q, flat_rank, args.ridge_scale)
        rows.append(
            {
                "flat_floor": float(flat),
                "curved_floor": float(curved),
                "drop": float(flat - curved),
                "curvature_proxy": float(curv),
            }
        )
        if (i + 1) % progress_stride == 0 or i + 1 == len(neighborhoods):
            log(f"{spec.label}: completed {i + 1}/{len(neighborhoods)} neighborhoods")
    flat_values = [row["flat_floor"] for row in rows]
    curved_values = [row["curved_floor"] for row in rows]
    drop_values = [row["drop"] for row in rows]
    curvature_values = [row["curvature_proxy"] for row in rows]
    return {
        "label": spec.label,
        "path": str(spec.path),
        "source_shape": [int(v) for v in np.load(spec.path, mmap_mode="r").shape],
        "sample_rows": int(x.shape[0]),
        "dimension": int(x.shape[1]),
        "sink_top_pc_fraction": sink_fraction,
        "tangent_rank": int(q),
        "quadratic_columns": int(quad_cols),
        "matched_flat_rank": int(flat_rank),
        "neighborhoods": int(len(rows)),
        "neighborhood_size": int(args.neighborhood_size),
        "train_fraction": float(args.train_fraction),
        "flat_floor_mean": float(np.mean(flat_values)),
        "flat_floor_median": float(np.median(flat_values)),
        "curved_floor_mean": float(np.mean(curved_values)),
        "curved_floor_median": float(np.median(curved_values)),
        "drop_mean": float(np.mean(drop_values)),
        "drop_median": float(np.median(drop_values)),
        "curvature_proxy_mean": float(np.mean(curvature_values)),
        "curvature_proxy_median": float(np.median(curvature_values)),
        "curvature_drop_correlation": correlation(curvature_values, drop_values),
        "neighborhood_results": rows,
    }


def write_report(out_dir: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Qwen Geometric-Wall Closure Test",
        "",
        "This run uses existing Qwen activation arrays and numpy-only local fits.",
        "Each layer is deterministically subsampled, centered, and peeled along its top PCA sink direction.",
        "",
        "Flat uses held-out local PCA with rank `q + q(q+1)/2`.",
        "Curved uses rank-`q` tangent PCA plus all centered quadratic tangent products.",
        "Both lanes therefore have the same output-parameter budget per local neighborhood.",
        "",
        "## Settings",
        "",
        f"- sample rows per layer: {payload['settings']['sample_rows']}",
        f"- neighborhoods per layer: {payload['settings']['neighborhoods']}",
        f"- neighborhood size: {payload['settings']['neighborhood_size']}",
        f"- tangent rank q: {payload['settings']['tangent_rank']}",
        f"- matched flat rank: {payload['settings']['matched_flat_rank']}",
        f"- train fraction: {payload['settings']['train_fraction']}",
        f"- ridge scale: {payload['settings']['ridge_scale']}",
        "",
        "## Layer Results",
        "",
        "| layer | sink frac | flat floor | curved floor | drop | curvature proxy | curvature/drop r |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["layers"]:
        corr = row["curvature_drop_correlation"]
        corr_text = "null" if corr is None else f"{corr:.6f}"
        lines.append(
            "| {label} | {sink_top_pc_fraction:.6f} | {flat_floor_mean:.6f} | "
            "{curved_floor_mean:.6f} | {drop_mean:.6f} | "
            "{curvature_proxy_mean:.6f} | ".format(**row)
            + corr_text
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Floors are held-out residual-energy fractions in post-sink local neighborhoods.",
            "- Drop is `flat_floor_mean - curved_floor_mean`; positive values mean the curved chart lowers the floor.",
            "- The curvature proxy is the RMS quadratic correction energy divided by held-out local energy.",
            "- The reported correlation is across neighborhoods within each layer.",
        ]
    )
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", action="append", required=True, help="LABEL:PATH activation npy")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample-rows", type=int, default=30_000)
    ap.add_argument("--neighborhoods", type=int, default=24)
    ap.add_argument("--neighborhood-size", type=int, default=240)
    ap.add_argument("--neighbor-dims", type=int, default=32)
    ap.add_argument("--tangent-rank", type=int, default=10)
    ap.add_argument("--train-fraction", type=float, default=0.75)
    ap.add_argument("--ridge-scale", type=float, default=1e-6)
    ap.add_argument("--pca-oversample", type=int, default=24)
    ap.add_argument("--power-iter", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1729)
    args = ap.parse_args()

    specs = parse_layer_specs(args.layer)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q = args.tangent_rank
    matched_flat_rank = q + q * (q + 1) // 2
    payload = {  # type: Dict[str, Any]
        "experiment": "qwen_wall_closure",
        "settings": {
            "sample_rows": int(args.sample_rows),
            "neighborhoods": int(args.neighborhoods),
            "neighborhood_size": int(args.neighborhood_size),
            "neighbor_dims": int(args.neighbor_dims),
            "tangent_rank": int(q),
            "matched_flat_rank": int(matched_flat_rank),
            "train_fraction": float(args.train_fraction),
            "ridge_scale": float(args.ridge_scale),
            "pca_oversample": int(args.pca_oversample),
            "power_iter": int(args.power_iter),
            "seed": int(args.seed),
        },
        "layers": [],
    }
    for layer_index, spec in enumerate(specs):
        payload["layers"].append(layer_experiment(args, spec, layer_index))
    with (out_dir / "numbers.json").open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    write_report(out_dir, payload)
    print("RESULTS_JSON " + json.dumps(payload))


if __name__ == "__main__":
    main()
