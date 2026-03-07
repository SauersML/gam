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
        default=REPO_ROOT / "scripts" / "duchon_2d_surface_order0_power2.png",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-train", type=int, default=900)
    parser.add_argument("--grid-size", type=int, default=90)
    return parser.parse_args()


def true_surface(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    # Smooth background plus a rotated 2D mesa: the surface rises up, stays high
    # over a bounded region, then drops back down in both rotated directions.
    u = 0.82 * x + 0.58 * z
    v = -0.46 * x + 0.89 * z

    smooth_bg = (
        0.06 * np.sin(2.0 * math.pi * u)
        + 0.04 * np.cos(2.0 * math.pi * v)
        + 0.03 * np.sin(math.pi * (x - 1.2 * z))
    )
    u_left = u - 0.34 + 0.04 * np.sin(1.6 * math.pi * v)
    u_right = u - 0.80 + 0.03 * np.cos(1.9 * math.pi * v - 0.4)
    v_low = v + 0.26 + 0.03 * np.sin(1.7 * math.pi * u + 0.2)
    v_high = v - 0.31 + 0.03 * np.cos(1.4 * math.pi * u - 0.3)

    gate_u = (1.0 / (1.0 + np.exp(-u_left / 0.026))) - (1.0 / (1.0 + np.exp(-u_right / 0.028)))
    gate_v = (1.0 / (1.0 + np.exp(-v_low / 0.03))) - (1.0 / (1.0 + np.exp(-v_high / 0.03)))
    plateau_gate = gate_u * gate_v
    plateau_base = 1.18 * plateau_gate
    plateau_variation = plateau_gate * (
        0.07 * np.sin(2.0 * math.pi * u + 0.15)
        + 0.06 * np.cos(2.3 * math.pi * v - 0.1)
    )
    return smooth_bg + plateau_base + plateau_variation


def write_csv(path: Path, rows: list[tuple[float, ...]], header: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def run(cmd: list[str | Path]) -> None:
    subprocess.run([str(x) for x in cmd], cwd=REPO_ROOT, check=True)


def build_demo_data(
    workdir: Path, seed: int, n_train: int, grid_size: int
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
    n_total = n_train
    x_all = rng.uniform(0.0, 1.0, size=n_total)
    z_all = rng.uniform(0.0, 1.0, size=n_total)
    y_all = true_surface(x_all, z_all) + rng.normal(0.0, 0.08, size=n_total)
    perm = rng.permutation(n_total)
    n_train_split = int(round(0.8 * n_total))
    train_idx = perm[:n_train_split]
    test_idx = perm[n_train_split:]
    x_train = x_all[train_idx]
    z_train = z_all[train_idx]
    y_train = y_all[train_idx]
    x_test = x_all[test_idx]
    z_test = z_all[test_idx]
    y_test = y_all[test_idx]

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
                "fit <- gam(y ~ s(x, z, bs='ds', m=c(1,0), k=min(31, nrow(train_df)-1)), data=train_df, method='REML', select=TRUE)",
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
    formula = "y ~ s(x, z, type=duchon, centers=31, order=0, power=2)"
    run(
        [
            gam_bin,
            "fit",
            train_csv,
            formula,
            "--adaptive-regularization",
            "true" if cfg.adaptive_regularization else "false",
            "--no-summary",
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


def main() -> int:
    args = parse_args()
    if not args.gam_bin.is_file():
        raise FileNotFoundError(f"gam binary not found at {args.gam_bin}")

    with tempfile.TemporaryDirectory(prefix="duchon_2d_surface_") as tmpdir:
        workdir = Path(tmpdir)
        train_csv, test_csv, grid_csv, x_train, z_train, y_train, x_test, z_test, y_test, xx, zz, y_true = build_demo_data(
            workdir, args.seed, args.n_train, args.grid_size
        )
        expected_rows = args.grid_size * args.grid_size
        grid_shape = xx.shape
        surfaces: list[tuple[str, np.ndarray, str, float]] = []
        skipped: list[str] = []
        n_test_rows = len(y_test)
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
            surfaces.append((cfg.label, y_hat, cfg.color, r2_score(y_test, y_test_hat)))
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
            surfaces.append(("mgcv duchon", y_mgcv, "#edae49", r2_score(y_test, y_mgcv_test)))
        except subprocess.CalledProcessError as exc:
            skipped.append(f"mgcv duchon (exit {exc.returncode})")
        if len(surfaces) != 3:
            raise RuntimeError(f"expected 3 fitted surfaces, got {len(surfaces)}; skipped={skipped}")
        make_plot(args.out, x_train, z_train, y_train, x_test, z_test, y_test, xx, zz, y_true, surfaces)

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
