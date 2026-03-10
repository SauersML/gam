#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAM_BIN = REPO_ROOT / "target" / "release" / "gam"
SMOOTH_RESOLUTION = 150


@dataclass(frozen=True)
class RustDuchonSurfaceConfig:
    label: str
    adaptive_regularization: bool
    color: str


CONFIGS: list[RustDuchonSurfaceConfig] = [
    RustDuchonSurfaceConfig(
        label="adaptive order=0 power=2",
        adaptive_regularization=True,
        color="#d1495b",
    ),
    RustDuchonSurfaceConfig(
        label="non-adaptive order=0 power=2",
        adaptive_regularization=False,
        color="#118ab2",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a 2D Duchon surface with a sharp transition to a new plateau, "
            "comparing adaptive vs non-adaptive regularization."
        )
    )
    parser.add_argument("--gam-bin", type=Path, default=DEFAULT_GAM_BIN)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "scripts" / "duchon_2d.png",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-train", type=int, default=4320)
    parser.add_argument("--n-test", type=int, default=10800)
    parser.add_argument("--grid-size", type=int, default=90)
    return parser.parse_args()


def true_surface(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    def sigmoid(t: np.ndarray, scale: float) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-t / scale))

    def rotated_bump(
        x0: float,
        z0: float,
        amp: float,
        sx: float,
        sz: float,
        theta: float,
    ) -> np.ndarray:
        dx = x - x0
        dz = z - z0
        ct = math.cos(theta)
        st = math.sin(theta)
        xr = ct * dx + st * dz
        zr = -st * dx + ct * dz
        return amp * np.exp(-0.5 * ((xr / sx) ** 2 + (zr / sz) ** 2))

    # Large-scale regions: mostly flat, with a few broad plateaus and basins.
    region_a = (sigmoid(x - 0.18, 0.025) - sigmoid(x - 0.42, 0.025)) * (
        sigmoid(z - 0.12, 0.03) - sigmoid(z - 0.36, 0.03)
    )
    region_b = (sigmoid(x - 0.56, 0.03) - sigmoid(x - 0.90, 0.03)) * (
        sigmoid(z - 0.58, 0.03) - sigmoid(z - 0.88, 0.03)
    )
    region_c = (sigmoid(x - 0.48, 0.028) - sigmoid(x - 0.72, 0.028)) * (
        sigmoid(z - 0.20, 0.028) - sigmoid(z - 0.44, 0.028)
    )
    flat_regions = 0.65 * region_a - 0.42 * region_b + 0.28 * region_c

    # Keep the background gentle so the flat regions still read as flat.
    background = (
        0.025 * np.sin(2.0 * math.pi * (0.7 * x + 0.2 * z))
        + 0.02 * np.cos(2.0 * math.pi * (-0.25 * x + 0.65 * z))
    )

    # Many localized spikes with varying sign, scale, and orientation.
    spikes = (
        rotated_bump(0.12, 0.18, 0.95, 0.020, 0.050, 0.35)
        + rotated_bump(0.20, 0.74, 0.72, 0.030, 0.022, -0.85)
        + rotated_bump(0.28, 0.46, -0.55, 0.050, 0.018, 0.55)
        + rotated_bump(0.34, 0.86, 1.15, 0.018, 0.040, 1.05)
        + rotated_bump(0.41, 0.24, 0.58, 0.026, 0.026, 0.00)
        + rotated_bump(0.47, 0.63, -0.62, 0.040, 0.020, -0.30)
        + rotated_bump(0.55, 0.12, 0.88, 0.022, 0.060, 0.90)
        + rotated_bump(0.61, 0.49, 1.25, 0.015, 0.028, -1.10)
        + rotated_bump(0.68, 0.78, -0.48, 0.060, 0.022, 0.40)
        + rotated_bump(0.74, 0.30, 1.40, 0.020, 0.020, 0.00)
        + rotated_bump(0.81, 0.57, 0.66, 0.030, 0.017, -0.70)
        + rotated_bump(0.88, 0.18, -0.52, 0.045, 0.020, 0.20)
        + rotated_bump(0.84, 0.84, 0.92, 0.024, 0.045, -0.45)
    )

    return background + flat_regions + spikes


def write_csv(path: Path, rows: list[tuple[float, ...]], header: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def run(cmd: list[str | Path]) -> None:
    subprocess.run([str(x) for x in cmd], cwd=REPO_ROOT, check=True)


def build_demo_data(
    workdir: Path, seed: int, n_train: int, n_test: int, grid_size: int
) -> tuple[
    Path,
    Path,
    Path,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(0.0, 1.0, size=n_train)
    z_train = rng.uniform(0.0, 1.0, size=n_train)
    y_train = true_surface(x_train, z_train) + rng.normal(0.0, 0.08, size=n_train)
    x_test = rng.uniform(0.0, 1.0, size=n_test)
    z_test = rng.uniform(0.0, 1.0, size=n_test)
    y_test = true_surface(x_test, z_test) + rng.normal(0.0, 0.08, size=n_test)

    x_axis = np.linspace(0.0, 1.0, grid_size)
    z_axis = np.linspace(0.0, 1.0, grid_size)
    xx, zz = np.meshgrid(x_axis, z_axis, indexing="xy")
    y_grid = true_surface(xx, zz)

    train_csv = workdir / "train.csv"
    test_csv = workdir / "test.csv"
    grid_csv = workdir / "grid.csv"
    write_csv(
        train_csv,
        [(float(x), float(z), float(y)) for x, z, y in zip(x_train, z_train, y_train)],
        ["x", "z", "y"],
    )
    write_csv(
        test_csv,
        [(float(x), float(z), float(y)) for x, z, y in zip(x_test, z_test, y_test)],
        ["x", "z", "y"],
    )
    write_csv(
        grid_csv,
        [(float(x), float(z)) for x, z in zip(xx.ravel(), zz.ravel())],
        ["x", "z"],
    )
    return train_csv, test_csv, grid_csv, x_train, z_train, y_train, x_test, z_test, y_test, xx, zz, y_grid


def read_prediction_mean(path: Path, expected_rows: int, grid_shape: tuple[int, int]) -> np.ndarray:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        mean = [float(row["mean"]) for row in reader]
    if len(mean) != expected_rows:
        raise RuntimeError(
            f"prediction row mismatch for {path.name}: got {len(mean)}, expected {expected_rows}"
        )
    return np.asarray(mean).reshape(grid_shape)


def read_prediction_vector(path: Path, expected_rows: int) -> np.ndarray:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        mean = [float(row["mean"]) for row in reader]
    if len(mean) != expected_rows:
        raise RuntimeError(
            f"prediction row mismatch for {path.name}: got {len(mean)}, expected {expected_rows}"
        )
    return np.asarray(mean)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def fit_mgcv_surface(
    workdir: Path,
    train_csv: Path,
    test_csv: Path,
    grid_csv: Path,
    expected_rows: int,
    n_test_rows: int,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    pred_path = workdir / "mgcv_duchon_surface.csv"
    test_pred_path = workdir / "mgcv_duchon_surface_test.csv"
    r_script = workdir / "fit_mgcv_duchon_surface.R"
    r_script.write_text(
        "\n".join(
            [
                "suppressPackageStartupMessages(library(mgcv))",
                f"train_df <- read.csv({train_csv.as_posix()!r})",
                f"test_df <- read.csv({test_csv.as_posix()!r})",
                f"grid_df <- read.csv({grid_csv.as_posix()!r})",
                f"fit <- gam(y ~ s(x, z, bs='ds', m=c(1,0), k=min({SMOOTH_RESOLUTION}, nrow(train_df)-1)), data=train_df, method='REML', select=TRUE)",
                "pred_grid <- predict(fit, newdata=grid_df, type='response')",
                "pred_test <- predict(fit, newdata=test_df, type='response')",
                f"write.csv(data.frame(mean=pred_grid), file={pred_path.as_posix()!r}, row.names=FALSE)",
                f"write.csv(data.frame(mean=pred_test), file={test_pred_path.as_posix()!r}, row.names=FALSE)",
            ]
        )
        + "\n"
    )
    run(["Rscript", r_script])
    return (
        read_prediction_mean(pred_path, expected_rows, grid_shape),
        read_prediction_vector(test_pred_path, n_test_rows),
    )


def fit_rust_surface(
    gam_bin: Path,
    workdir: Path,
    train_csv: Path,
    test_csv: Path,
    grid_csv: Path,
    cfg: RustDuchonSurfaceConfig,
    expected_rows: int,
    n_test_rows: int,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    slug = cfg.label.lower().replace(" ", "_").replace("=", "").replace(",", "")
    model_path = workdir / f"{slug}.json"
    pred_path = workdir / f"{slug}.csv"
    test_pred_path = workdir / f"{slug}_test.csv"
    formula = f"y ~ s(x, z, type=duchon, centers={SMOOTH_RESOLUTION}, order=0, power=2)"
    run(
        [
            gam_bin,
            "fit",
            train_csv,
            formula,
            "--adaptive-regularization",
            "true" if cfg.adaptive_regularization else "false",
            "--out",
            model_path,
        ]
    )
    run([gam_bin, "predict", model_path, grid_csv, "--out", pred_path])
    run([gam_bin, "predict", model_path, test_csv, "--out", test_pred_path])
    return (
        read_prediction_mean(pred_path, expected_rows, grid_shape),
        read_prediction_vector(test_pred_path, n_test_rows),
    )


def make_plot(
    out_path: Path,
    x_train: np.ndarray,
    z_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    xx: np.ndarray,
    zz: np.ndarray,
    y_true: np.ndarray,
    surfaces: list[tuple[str, np.ndarray, str, float]],
) -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "path.simplify": False,
            "agg.path.chunksize": 0,
        }
    )
    fig = plt.figure(figsize=(21.0, 5.8), constrained_layout=True)
    fig.patch.set_facecolor("#f7f2ea")
    axes = [
        fig.add_subplot(1, 4, 1, projection="3d"),
        fig.add_subplot(1, 4, 2, projection="3d"),
        fig.add_subplot(1, 4, 3, projection="3d"),
        fig.add_subplot(1, 4, 4, projection="3d"),
    ]

    plotted = [("true surface", y_true, "Greys", r2_score(y_test, true_surface(x_test, z_test))), *[(label, y_hat, None, r2) for label, y_hat, _, r2 in surfaces]]
    color_lookup = {label: color for label, _, color, _ in surfaces}

    vmin = min(np.min(y_true), *(np.min(y_hat) for _, y_hat, _, _ in surfaces))
    vmax = max(np.max(y_true), *(np.max(y_hat) for _, y_hat, _, _ in surfaces))

    for ax, (label, yy, cmap_name, r2) in zip(axes, plotted):
        ax.set_facecolor("#fffaf3")
        if cmap_name is not None:
            ax.plot_surface(
                xx,
                zz,
                yy,
                cmap=cmap_name,
                linewidth=0,
                antialiased=True,
                alpha=0.95,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            ax.plot_surface(
                xx,
                zz,
                yy,
                color=color_lookup[label],
                linewidth=0,
                antialiased=True,
                alpha=0.9,
                shade=True,
            )
        ax.scatter(
            x_train,
            z_train,
            y_train,
            s=8,
            color="#3b3b3b",
            alpha=0.16,
            depthshade=False,
        )
        ax.scatter(
            x_test,
            z_test,
            y_test,
            s=10,
            color="#111111",
            alpha=0.22,
            depthshade=False,
        )
        ax.view_init(elev=28, azim=-130)
        ax.set_title(f"{label}\n$R^2_{{test}}$ = {r2:.3f}", pad=12)
        ax.set_xlabel("x", labelpad=8)
        ax.set_ylabel("z", labelpad=8)
        ax.set_zlabel("mean", labelpad=8)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(vmin, vmax)
        ax.xaxis.pane.set_facecolor((1.0, 0.98, 0.95, 1.0))
        ax.yaxis.pane.set_facecolor((1.0, 0.98, 0.95, 1.0))
        ax.zaxis.pane.set_facecolor((1.0, 0.98, 0.95, 1.0))

    fig.suptitle("2D Duchon Surface: Order 0, Power 2", fontsize=16, color="#2f2925")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def print_metrics_report(metrics: list[tuple[str, float, float, float]]) -> None:
    print("test-set diagnostics:")
    for label, r2, rmse_val, mae_val in metrics:
        print(
            f"- {label}: "
            f"R^2={r2:.4f}, "
            f"RMSE={rmse_val:.4f}, "
            f"MAE={mae_val:.4f}"
        )


def main() -> int:
    args = parse_args()
    if not args.gam_bin.is_file():
        raise FileNotFoundError(f"gam binary not found at {args.gam_bin}")

    with tempfile.TemporaryDirectory(prefix="duchon_2d_surface_") as tmpdir:
        workdir = Path(tmpdir)
        train_csv, test_csv, grid_csv, x_train, z_train, y_train, x_test, z_test, y_test, xx, zz, y_true = build_demo_data(
            workdir, args.seed, args.n_train, args.n_test, args.grid_size
        )
        expected_rows = args.grid_size * args.grid_size
        grid_shape = xx.shape
        surfaces: list[tuple[str, np.ndarray, str, float]] = []
        metrics: list[tuple[str, float, float, float]] = []
        skipped: list[str] = []
        n_test_rows = len(y_test)
        true_r2 = r2_score(y_test, true_surface(x_test, z_test))
        true_rmse = rmse(y_test, true_surface(x_test, z_test))
        true_mae = mae(y_test, true_surface(x_test, z_test))
        metrics.append(("true surface", true_r2, true_rmse, true_mae))
        for cfg in CONFIGS:
            try:
                y_hat, y_test_hat = fit_rust_surface(
                    args.gam_bin,
                    workdir,
                    train_csv,
                    test_csv,
                    grid_csv,
                    cfg,
                    expected_rows,
                    n_test_rows,
                    grid_shape,
                )
            except subprocess.CalledProcessError as exc:
                skipped.append(f"{cfg.label} (exit {exc.returncode})")
                continue
            r2 = r2_score(y_test, y_test_hat)
            rmse_val = rmse(y_test, y_test_hat)
            mae_val = mae(y_test, y_test_hat)
            surfaces.append((cfg.label, y_hat, cfg.color, r2))
            metrics.append((cfg.label, r2, rmse_val, mae_val))
        try:
            y_mgcv, y_mgcv_test = fit_mgcv_surface(
                workdir,
                train_csv,
                test_csv,
                grid_csv,
                expected_rows,
                n_test_rows,
                grid_shape,
            )
            r2 = r2_score(y_test, y_mgcv_test)
            rmse_val = rmse(y_test, y_mgcv_test)
            mae_val = mae(y_test, y_mgcv_test)
            surfaces.append(("mgcv duchon", y_mgcv, "#edae49", r2))
            metrics.append(("mgcv duchon", r2, rmse_val, mae_val))
        except subprocess.CalledProcessError as exc:
            skipped.append(f"mgcv duchon (exit {exc.returncode})")
        if len(surfaces) != 3:
            raise RuntimeError(f"expected 3 fitted surfaces, got {len(surfaces)}; skipped={skipped}")
        make_plot(args.out, x_train, z_train, y_train, x_test, z_test, y_test, xx, zz, y_true, surfaces)
        print_metrics_report(metrics)

    if skipped:
        print("skipped configs:")
        for item in skipped:
            print(f"- {item}")
    print(args.out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        raise
