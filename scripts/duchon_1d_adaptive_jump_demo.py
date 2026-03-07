#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAM_BIN = REPO_ROOT / "target" / "release" / "gam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 1D Duchon demo with a sharp middle jump, fit with "
            "adaptive regularization on/off, and save a single comparison PNG."
        )
    )
    parser.add_argument(
        "--gam-bin",
        type=Path,
        default=DEFAULT_GAM_BIN,
        help=f"Path to the gam CLI binary (default: {DEFAULT_GAM_BIN})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "scripts" / "duchon_1d_adaptive_jump_demo.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for synthetic data generation",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=320,
        help="Number of training samples",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=50000,
        help="Number of prediction grid points",
    )
    return parser.parse_args()


def true_curve(x: np.ndarray) -> np.ndarray:
    smooth_background = 0.12 * np.sin(2.0 * math.pi * x) + 0.05 * np.cos(5.0 * math.pi * x)
    sharp_mid_jump = 1.4 / (1.0 + np.exp(-(x - 0.5) / 0.012))
    return smooth_background + sharp_mid_jump


def write_csv(path: Path, rows: list[tuple[float, ...]], header: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def run(cmd: list[str | Path]) -> None:
    rendered = [str(part) for part in cmd]
    subprocess.run(rendered, cwd=REPO_ROOT, check=True)


def build_demo_data(
    workdir: Path,
    seed: int,
    n_train: int,
    n_grid: int,
) -> tuple[Path, Path, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train = np.sort(rng.uniform(0.0, 1.0, size=n_train))
    y_train = true_curve(x_train) + rng.normal(0.0, 0.09, size=n_train)
    x_grid = np.linspace(0.0, 1.0, n_grid)
    y_grid = true_curve(x_grid)

    train_csv = workdir / "train.csv"
    grid_csv = workdir / "grid.csv"
    write_csv(train_csv, [(float(x), float(y)) for x, y in zip(x_train, y_train)], ["x", "y"])
    write_csv(grid_csv, [(float(x),) for x in x_grid], ["x"])
    return train_csv, grid_csv, x_train, y_train, x_grid, y_grid


def fit_and_predict(
    gam_bin: Path,
    workdir: Path,
    train_csv: Path,
    grid_csv: Path,
    adaptive_regularization: bool,
    expected_rows: int,
) -> np.ndarray:
    label = "true" if adaptive_regularization else "false"
    model_path = workdir / f"duchon_1d_adaptive_{label}.json"
    pred_path = workdir / f"duchon_1d_adaptive_{label}.csv"
    formula = "y ~ s(x, type=duchon, centers=31)"

    run(
        [
            gam_bin,
            "fit",
            train_csv,
            formula,
            "--adaptive-regularization",
            label,
            "--no-summary",
            "--out",
            model_path,
        ]
    )
    run([gam_bin, "predict", model_path, grid_csv, "--out", pred_path])

    with pred_path.open() as handle:
        reader = csv.DictReader(handle)
        mean = [float(row["mean"]) for row in reader]
    if len(mean) != expected_rows:
        raise RuntimeError(
            f"prediction row mismatch for adaptive_regularization={label}: "
            f"got {len(mean)}, expected {expected_rows}"
        )
    return np.asarray(mean)


def fit_and_predict_rust_pspline(
    gam_bin: Path,
    workdir: Path,
    train_csv: Path,
    grid_csv: Path,
    expected_rows: int,
) -> np.ndarray:
    model_path = workdir / "rust_pspline.json"
    pred_path = workdir / "rust_pspline.csv"
    formula = "y ~ s(x, type=ps, knots=31)"

    run(
        [
            gam_bin,
            "fit",
            train_csv,
            formula,
            "--no-summary",
            "--out",
            model_path,
        ]
    )
    run([gam_bin, "predict", model_path, grid_csv, "--out", pred_path])

    with pred_path.open() as handle:
        reader = csv.DictReader(handle)
        mean = [float(row["mean"]) for row in reader]
    if len(mean) != expected_rows:
        raise RuntimeError(
            f"rust p-spline prediction row mismatch: got {len(mean)}, expected {expected_rows}"
        )
    return np.asarray(mean)


def fit_and_predict_rust_tps(
    gam_bin: Path,
    workdir: Path,
    train_csv: Path,
    grid_csv: Path,
    expected_rows: int,
) -> np.ndarray:
    model_path = workdir / "rust_tps.json"
    pred_path = workdir / "rust_tps.csv"
    formula = "y ~ s(x, type=tps, centers=31)"

    run(
        [
            gam_bin,
            "fit",
            train_csv,
            formula,
            "--no-summary",
            "--out",
            model_path,
        ]
    )
    run([gam_bin, "predict", model_path, grid_csv, "--out", pred_path])

    with pred_path.open() as handle:
        reader = csv.DictReader(handle)
        mean = [float(row["mean"]) for row in reader]
    if len(mean) != expected_rows:
        raise RuntimeError(
            f"rust TPS prediction row mismatch: got {len(mean)}, expected {expected_rows}"
        )
    return np.asarray(mean)


def fit_and_predict_rust_matern(
    gam_bin: Path,
    workdir: Path,
    train_csv: Path,
    grid_csv: Path,
    expected_rows: int,
) -> np.ndarray:
    model_path = workdir / "rust_matern.json"
    pred_path = workdir / "rust_matern.csv"
    formula = "y ~ s(x, type=matern, centers=31)"

    run(
        [
            gam_bin,
            "fit",
            train_csv,
            formula,
            "--no-summary",
            "--out",
            model_path,
        ]
    )
    run([gam_bin, "predict", model_path, grid_csv, "--out", pred_path])

    with pred_path.open() as handle:
        reader = csv.DictReader(handle)
        mean = [float(row["mean"]) for row in reader]
    if len(mean) != expected_rows:
        raise RuntimeError(
            f"rust Matérn prediction row mismatch: got {len(mean)}, expected {expected_rows}"
        )
    return np.asarray(mean)


def fit_and_predict_mgcv_duchon(
    workdir: Path,
    train_csv: Path,
    grid_csv: Path,
    expected_rows: int,
) -> np.ndarray:
    return fit_and_predict_mgcv(
        workdir,
        train_csv,
        grid_csv,
        expected_rows,
        label="duchon",
        smooth_term="s(x, bs='ds', m=c(1,0), k=min(31, nrow(train_df)-1))",
    )


def fit_and_predict_mgcv(
    workdir: Path,
    train_csv: Path,
    grid_csv: Path,
    expected_rows: int,
    *,
    label: str,
    smooth_term: str,
) -> np.ndarray:
    pred_path = workdir / f"mgcv_{label}.csv"
    r_script = workdir / f"fit_mgcv_{label}.R"
    r_script.write_text(
        "\n".join(
            [
                "suppressPackageStartupMessages(library(mgcv))",
                f"train_df <- read.csv({train_csv.as_posix()!r})",
                f"grid_df <- read.csv({grid_csv.as_posix()!r})",
                f"fit <- gam(y ~ {smooth_term}, data=train_df, method='REML', select=TRUE)",
                "pred <- predict(fit, newdata=grid_df, type='response')",
                f"write.csv(data.frame(mean=pred), file={pred_path.as_posix()!r}, row.names=FALSE)",
            ]
        )
        + "\n"
    )
    run(["Rscript", r_script])
    with pred_path.open() as handle:
        reader = csv.DictReader(handle)
        mean = [float(row["mean"]) for row in reader]
    if len(mean) != expected_rows:
        raise RuntimeError(
            f"mgcv {label} prediction row mismatch: got {len(mean)}, expected {expected_rows}"
        )
    return np.asarray(mean)


def fit_and_predict_mgcv_matern(
    workdir: Path,
    train_csv: Path,
    grid_csv: Path,
    expected_rows: int,
) -> np.ndarray:
    return fit_and_predict_mgcv(
        workdir,
        train_csv,
        grid_csv,
        expected_rows,
        label="matern",
        smooth_term="s(x, bs='gp', m=c(-4,1.0), k=min(31, nrow(train_df)-1))",
    )


def fit_and_predict_mgcv_pspline(
    workdir: Path,
    train_csv: Path,
    grid_csv: Path,
    expected_rows: int,
) -> np.ndarray:
    return fit_and_predict_mgcv(
        workdir,
        train_csv,
        grid_csv,
        expected_rows,
        label="pspline",
        smooth_term="s(x, bs='ps', k=min(35, nrow(train_df)-1))",
    )


def fit_and_predict_mgcv_tps(
    workdir: Path,
    train_csv: Path,
    grid_csv: Path,
    expected_rows: int,
) -> np.ndarray:
    return fit_and_predict_mgcv(
        workdir,
        train_csv,
        grid_csv,
        expected_rows,
        label="tps",
        smooth_term="s(x, bs='tp', k=min(31, nrow(train_df)-1))",
    )


def make_plot(
    out_path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_grid: np.ndarray,
    y_true: np.ndarray,
    y_adaptive_true: np.ndarray,
    y_adaptive_false: np.ndarray,
    y_mgcv: np.ndarray,
    y_mgcv_matern: np.ndarray,
    y_mgcv_pspline: np.ndarray,
    y_mgcv_tps: np.ndarray,
    y_rust_pspline: np.ndarray,
    y_rust_tps: np.ndarray,
    y_rust_matern: np.ndarray,
) -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "path.simplify": False,
            "agg.path.chunksize": 0,
        }
    )
    fig, ax = plt.subplots(figsize=(11.0, 6.6), constrained_layout=True)
    fig.patch.set_facecolor("#f7f2ea")
    ax.set_facecolor("#fffaf3")
    ax.scatter(
        x_train,
        y_train,
        s=18,
        alpha=0.24,
        color="#6c757d",
        edgecolors="none",
        label="training data",
    )
    ax.plot(
        x_grid,
        y_true,
        color="#2b2b2b",
        linewidth=2.0,
        linestyle=(0, (1.2, 2.4)),
        alpha=0.9,
        label="true curve",
    )
    ax.plot(
        x_grid,
        y_adaptive_true,
        color="#d1495b",
        linewidth=3.0,
        alpha=0.78,
        label="--adaptive-regularization true",
    )
    ax.plot(
        x_grid,
        y_adaptive_false,
        color="#00798c",
        linewidth=3.0,
        alpha=0.78,
        label="--adaptive-regularization false",
    )
    ax.plot(
        x_grid,
        y_mgcv,
        color="#edae49",
        linewidth=2.8,
        alpha=0.82,
        label="mgcv duchon",
    )
    ax.plot(
        x_grid,
        y_mgcv_matern,
        color="#f4a261",
        linewidth=2.3,
        alpha=0.72,
        label="mgcv Matérn",
    )
    ax.plot(
        x_grid,
        y_mgcv_pspline,
        color="#2a9d8f",
        linewidth=2.3,
        alpha=0.72,
        label="mgcv p-spline",
    )
    ax.plot(
        x_grid,
        y_mgcv_tps,
        color="#264653",
        linewidth=2.3,
        alpha=0.72,
        label="mgcv TPS",
    )
    ax.plot(
        x_grid,
        y_rust_pspline,
        color="#4f772d",
        linewidth=2.6,
        alpha=0.8,
        label="rust p-spline",
    )
    ax.plot(
        x_grid,
        y_rust_tps,
        color="#5e548e",
        linewidth=2.5,
        alpha=0.78,
        label="rust TPS",
    )
    ax.plot(
        x_grid,
        y_rust_matern,
        color="#bc6c25",
        linewidth=2.5,
        alpha=0.78,
        label="rust Matérn",
    )
    ax.axvline(0.5, color="#8a817c", linewidth=1.2, linestyle="--", alpha=0.45)
    ax.set_title("1D Duchon Fit With and Without Adaptive Regularization", pad=16)
    ax.set_xlabel("x")
    ax.set_ylabel("fitted mean")
    ax.legend(
        frameon=True,
        facecolor="#fffaf3",
        edgecolor="#eadfce",
        framealpha=0.95,
        loc="upper left",
    )
    ax.grid(color="#d8cfc3", alpha=0.28, linewidth=0.8)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#7d746d")
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(colors="#534b45")
    ax.xaxis.label.set_color("#3a332f")
    ax.yaxis.label.set_color("#3a332f")
    ax.title.set_color("#2f2925")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.gam_bin.is_file():
        raise FileNotFoundError(
            f"gam binary not found at {args.gam_bin}. Build it first with `cargo build --release --bin gam`."
        )

    with tempfile.TemporaryDirectory(prefix="duchon_1d_adaptive_demo_") as tmpdir:
        workdir = Path(tmpdir)
        train_csv, grid_csv, x_train, y_train, x_grid, y_true = build_demo_data(
            workdir,
            args.seed,
            args.n_train,
            args.n_grid,
        )
        y_adaptive_true = fit_and_predict(
            args.gam_bin,
            workdir,
            train_csv,
            grid_csv,
            adaptive_regularization=True,
            expected_rows=args.n_grid,
        )
        y_adaptive_false = fit_and_predict(
            args.gam_bin,
            workdir,
            train_csv,
            grid_csv,
            adaptive_regularization=False,
            expected_rows=args.n_grid,
        )
        y_mgcv = fit_and_predict_mgcv_duchon(
            workdir,
            train_csv,
            grid_csv,
            expected_rows=args.n_grid,
        )
        y_mgcv_matern = fit_and_predict_mgcv_matern(
            workdir,
            train_csv,
            grid_csv,
            expected_rows=args.n_grid,
        )
        y_mgcv_pspline = fit_and_predict_mgcv_pspline(
            workdir,
            train_csv,
            grid_csv,
            expected_rows=args.n_grid,
        )
        y_mgcv_tps = fit_and_predict_mgcv_tps(
            workdir,
            train_csv,
            grid_csv,
            expected_rows=args.n_grid,
        )
        y_rust_pspline = fit_and_predict_rust_pspline(
            args.gam_bin,
            workdir,
            train_csv,
            grid_csv,
            expected_rows=args.n_grid,
        )
        y_rust_tps = fit_and_predict_rust_tps(
            args.gam_bin,
            workdir,
            train_csv,
            grid_csv,
            expected_rows=args.n_grid,
        )
        y_rust_matern = fit_and_predict_rust_matern(
            args.gam_bin,
            workdir,
            train_csv,
            grid_csv,
            expected_rows=args.n_grid,
        )
        if len(x_grid) != args.n_grid or len(y_true) != args.n_grid:
            raise RuntimeError(
                f"grid construction mismatch: len(x_grid)={len(x_grid)}, "
                f"len(y_true)={len(y_true)}, expected {args.n_grid}"
            )
        make_plot(
            args.out,
            x_train,
            y_train,
            x_grid,
            y_true,
            y_adaptive_true,
            y_adaptive_false,
            y_mgcv,
            y_mgcv_matern,
            y_mgcv_pspline,
            y_mgcv_tps,
            y_rust_pspline,
            y_rust_tps,
            y_rust_matern,
        )

    print(args.out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        raise
