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
class FitSpec:
    engine: str
    power: int
    color: str
    linestyle: str

    @property
    def label(self) -> str:
        return f"{self.engine} power={self.power}"


FIT_SPECS: list[FitSpec] = [
    FitSpec(engine="mgcv", power=0, color="#6A4C93", linestyle="-"),
    FitSpec(engine="mgcv", power=1, color="#00798C", linestyle="-"),
    FitSpec(engine="mgcv", power=2, color="#D1495B", linestyle="-"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 1D Duchon fits across mgcv and Rust for power=1 and power=2 "
            "using the same synthetic training data."
        )
    )
    parser.add_argument("--gam-bin", type=Path, default=DEFAULT_GAM_BIN)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "scripts" / "duchon_1d_power_compare.png",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-train", type=int, default=320)
    parser.add_argument("--n-grid", type=int, default=4000)
    parser.add_argument("--centers", type=int, default=31)
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
    subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT, check=True)


def build_demo_data(
    workdir: Path, seed: int, n_train: int, n_grid: int
) -> tuple[Path, Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def read_prediction_mean(path: Path, expected_rows: int) -> np.ndarray:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        mean = [float(row["mean"]) for row in reader]
    if len(mean) != expected_rows:
        raise RuntimeError(
            f"prediction row mismatch for {path.name}: got {len(mean)}, expected {expected_rows}"
        )
    return np.asarray(mean)


def fit_rust_curve(
    gam_bin: Path,
    workdir: Path,
    train_csv: Path,
    grid_csv: Path,
    expected_rows: int,
    centers: int,
    power: int,
) -> np.ndarray:
    slug = f"rust_power{power}"
    model_path = workdir / f"{slug}.json"
    pred_path = workdir / f"{slug}.csv"
    formula = f"y ~ s(x, type=duchon, centers={centers}, order=0, power={power})"
    run(
        [
            gam_bin,
            "fit",
            train_csv,
            formula,
            "--adaptive-regularization",
            "false",
            "--out",
            model_path,
        ]
    )
    run([gam_bin, "predict", model_path, grid_csv, "--out", pred_path])
    return read_prediction_mean(pred_path, expected_rows)


def fit_mgcv_curve(
    workdir: Path,
    train_csv: Path,
    grid_csv: Path,
    expected_rows: int,
    centers: int,
    power: int,
) -> np.ndarray:
    slug = f"mgcv_power{power}"
    pred_path = workdir / f"{slug}.csv"
    r_script = workdir / f"fit_{slug}.R"
    r_script.write_text(
        "\n".join(
            [
                "suppressPackageStartupMessages(library(mgcv))",
                f"train_df <- read.csv({train_csv.as_posix()!r})",
                f"grid_df <- read.csv({grid_csv.as_posix()!r})",
                (
                    "fit <- gam("
                    f"y ~ s(x, bs='ds', m=c({power},0), k=min({centers}, nrow(train_df)-1)), "
                    "data=train_df, method='REML', select=TRUE)"
                ),
                "pred <- predict(fit, newdata=grid_df, type='response')",
                f"write.csv(data.frame(mean=pred), file={pred_path.as_posix()!r}, row.names=FALSE)",
            ]
        )
        + "\n"
    )
    run(["Rscript", r_script])
    return read_prediction_mean(pred_path, expected_rows)


def make_plot(
    out_path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_grid: np.ndarray,
    y_true: np.ndarray,
    curves: list[tuple[FitSpec, np.ndarray]],
    skipped: list[str],
) -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "path.simplify": False,
            "agg.path.chunksize": 0,
        }
    )
    power_values = sorted({spec.power for spec, _ in curves} | {spec.power for spec in FIT_SPECS})
    fig, axes = plt.subplots(1, len(power_values), figsize=(16.5, 5.8), constrained_layout=True, sharey=True)
    fig.patch.set_facecolor("#F7F2EA")
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    curve_lookup = {(spec.engine, spec.power): y_hat for spec, y_hat in curves}

    for ax, power in zip(axes, power_values):
        ax.set_facecolor("#FFF9F1")
        ax.scatter(
            x_train,
            y_train,
            s=16,
            alpha=0.20,
            color="#5C5550",
            edgecolors="none",
            label="training data",
        )
        ax.plot(
            x_grid,
            y_true,
            color="#1F1F1F",
            linewidth=2.0,
            linestyle=(0, (1.5, 2.2)),
            alpha=0.95,
            label="true curve",
        )
        for engine in ("mgcv",):
            spec = next(item for item in FIT_SPECS if item.engine == engine and item.power == power)
            y_hat = curve_lookup.get((engine, power))
            if y_hat is None:
                continue
            ax.plot(
                x_grid,
                y_hat,
                color=spec.color,
                linewidth=3.0 if spec.engine == "rust" else 2.4,
                linestyle=spec.linestyle,
                alpha=0.88,
                label=spec.label,
            )

        ax.axvline(0.5, color="#8A817C", linewidth=1.0, linestyle="--", alpha=0.45)
        ax.set_title(f"power={power}", pad=12)
        ax.set_xlabel("x")
        ax.grid(color="#D8CFC3", alpha=0.28, linewidth=0.8)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#7D746D")
            ax.spines[spine].set_linewidth(1.0)
        ax.tick_params(colors="#534B45")
        missing = [f"{engine} power={power}" for engine in ("mgcv",) if (engine, power) not in curve_lookup]
        if missing:
            ax.text(
                0.98,
                0.02,
                "missing: " + ", ".join(missing),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9.0,
                color="#6B625C",
                bbox={"facecolor": "#FFF9F1", "edgecolor": "#E9DDCA", "alpha": 0.92, "pad": 4},
            )

    axes[0].set_ylabel("fitted mean")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=True,
        facecolor="#FFF9F1",
        edgecolor="#E9DDCA",
        framealpha=0.96,
        loc="upper center",
        ncol=4,
    )
    fig.suptitle("1D Duchon Power Comparison: mgcv Only", color="#2F2925")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.gam_bin.is_file():
        raise FileNotFoundError(f"gam binary not found at {args.gam_bin}")

    with tempfile.TemporaryDirectory(prefix="duchon_1d_power_compare_") as tmpdir:
        workdir = Path(tmpdir)
        train_csv, grid_csv, x_train, y_train, x_grid, y_true = build_demo_data(
            workdir, args.seed, args.n_train, args.n_grid
        )
        curves: list[tuple[FitSpec, np.ndarray]] = []
        skipped: list[str] = []
        for spec in FIT_SPECS:
            try:
                if spec.engine == "rust":
                    y_hat = fit_rust_curve(
                        args.gam_bin,
                        workdir,
                        train_csv,
                        grid_csv,
                        args.n_grid,
                        args.centers,
                        spec.power,
                    )
                else:
                    y_hat = fit_mgcv_curve(
                        workdir,
                        train_csv,
                        grid_csv,
                        args.n_grid,
                        args.centers,
                        spec.power,
                    )
            except subprocess.CalledProcessError:
                skipped.append(spec.label)
                continue
            curves.append((spec, y_hat))

        make_plot(args.out, x_train, y_train, x_grid, y_true, curves, skipped)
        if skipped:
            print("skipped fits:")
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
