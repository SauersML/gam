#!/usr/bin/env python3
"""
Standalone figure generator for benchmark results.

Usage:
    python3 bench/generate_figures.py bench/results.nightly.json bench/figures bench/figures.zip

Reads a merged results JSON, generates one PNG per scenario, and bundles
them into a single .zip for easy download from GitHub Actions.
"""
from __future__ import annotations

import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Metric config per family
# ---------------------------------------------------------------------------

def _metric_display_config(family: str):
    if family == "binomial":
        return [
            ("auc", "AUC", True),
            ("logloss", "Log-Loss", False),
            ("brier", "Brier Score", False),
            ("nagelkerke_r2", "Nagelkerke R²", True),
        ]
    if family == "survival":
        return [
            ("auc", "C-Index", True),
            ("logloss", "Partial Log-Loss", False),
            ("brier", "Partial Brier", False),
            ("nagelkerke_r2", "Nagelkerke R²", True),
        ]
    return [
        ("rmse", "RMSE", False),
        ("r2", "R²", True),
        ("logloss", "Gaussian Log-Loss", False),
        ("mae", "MAE", False),
    ]


def _short_contender_label(name: str) -> str:
    return (
        name.replace("python_", "py·")
        .replace("r_mgcv_", "mgcv·")
        .replace("r_", "R·")
        .replace("rust_", "rust·")
        .replace("_", " ")
    )


# ---------------------------------------------------------------------------
# Theme constants (GitHub dark mode palette)
# ---------------------------------------------------------------------------

BG_DARK = "#0d1117"
BG_CARD = "#161b22"
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#21262d"
ACCENT_COLORS = [
    "#58a6ff", "#3fb950", "#d29922", "#f85149",
    "#bc8cff", "#79c0ff", "#56d364", "#e3b341",
    "#ff7b72", "#d2a8ff", "#a5d6ff", "#7ee787",
    "#f2cc60", "#ffa198", "#cabffd", "#b1bac4",
]


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_scenario_figures(results: list[dict], out_dir: Path) -> list[Path]:
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r.get("status") == "ok":
            by_scenario[r["scenario_name"]].append(r)

    paths: list[Path] = []
    for scenario_name, rows in sorted(by_scenario.items()):
        if not rows:
            continue
        family = rows[0].get("family", "gaussian")
        metrics_cfg = _metric_display_config(family)
        active_metrics = [
            (k, l, h) for k, l, h in metrics_cfg
            if any(r.get(k) is not None for r in rows)
        ]
        if not active_metrics:
            continue

        n_metrics = len(active_metrics)
        contenders = [r["contender"] for r in rows]
        n_contenders = len(contenders)

        fig_height = max(3.2, 1.2 + 0.42 * n_contenders) * (n_metrics + 1)
        fig, axes = plt.subplots(
            n_metrics + 1, 1,
            figsize=(10, min(fig_height, 28)),
            facecolor=BG_DARK,
            gridspec_kw={"hspace": 0.45},
        )
        if n_metrics + 1 == 1:
            axes = [axes]

        for ax in axes:
            ax.set_facecolor(BG_CARD)
            ax.tick_params(colors=TEXT_COLOR, which="both")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color(GRID_COLOR)
            ax.spines["left"].set_color(GRID_COLOR)
            ax.xaxis.label.set_color(TEXT_COLOR)
            ax.yaxis.label.set_color(TEXT_COLOR)
            ax.title.set_color(TEXT_COLOR)

        short_labels = [_short_contender_label(c) for c in contenders]
        y_pos = list(range(n_contenders))
        colors = [ACCENT_COLORS[i % len(ACCENT_COLORS)] for i in range(n_contenders)]

        # ---- Metric subplots ----
        for idx, (key, label, higher_is_better) in enumerate(active_metrics):
            ax = axes[idx]
            vals = [float(r.get(key, 0) or 0) for r in rows]
            bars = ax.barh(
                y_pos, vals, height=0.62,
                color=colors, edgecolor="none", alpha=0.88, zorder=3,
            )
            valid_vals = [v for v in vals if v != 0.0]
            if valid_vals:
                best_val = max(valid_vals) if higher_is_better else min(valid_vals)
                for bar, v in zip(bars, vals):
                    if abs(v - best_val) < 1e-12 and v != 0.0:
                        bar.set_edgecolor("#f0f6fc")
                        bar.set_linewidth(1.8)
                        bar.set_alpha(1.0)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(short_labels, fontsize=8.5)
            ax.invert_yaxis()
            direction = "↑ higher is better" if higher_is_better else "↓ lower is better"
            ax.set_title(f"{label}  ({direction})", fontsize=10, loc="left", pad=8)
            ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5, zorder=0)

            for bar, v in zip(bars, vals):
                if v == 0.0:
                    continue
                if abs(v) < 1.0:
                    fmt = f"{v:.4f}"
                elif abs(v) < 100:
                    fmt = f"{v:.3f}"
                else:
                    fmt = f"{v:.1f}"
                ax.text(
                    bar.get_width() + ax.get_xlim()[1] * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    fmt, va="center", ha="left",
                    fontsize=7.5, color=TEXT_COLOR, alpha=0.85,
                )

        # ---- Timing subplot ----
        ax_time = axes[-1]
        fit_times = [float(r.get("fit_sec", 0)) for r in rows]
        pred_times = [float(r.get("predict_sec", 0)) for r in rows]
        bars_fit = ax_time.barh(
            y_pos, fit_times, height=0.42, label="Fit",
            color="#58a6ff", alpha=0.8, zorder=3,
        )
        bars_pred = ax_time.barh(
            [y + 0.42 for y in y_pos], pred_times, height=0.42, label="Predict",
            color="#3fb950", alpha=0.8, zorder=3,
        )
        ax_time.set_yticks([y + 0.21 for y in y_pos])
        ax_time.set_yticklabels(short_labels, fontsize=8.5)
        ax_time.invert_yaxis()
        ax_time.set_title("Fit + Predict Time  (↓ lower is better)", fontsize=10, loc="left", pad=8)
        ax_time.grid(axis="x", color=GRID_COLOR, linewidth=0.5, zorder=0)
        ax_time.legend(
            loc="lower right", fontsize=8,
            facecolor=BG_CARD, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
        )
        for bar, v in zip(bars_fit, fit_times):
            if v > 0:
                ax_time.text(
                    bar.get_width() + ax_time.get_xlim()[1] * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}s", va="center", ha="left",
                    fontsize=7, color=TEXT_COLOR, alpha=0.75,
                )
        for bar, v in zip(bars_pred, pred_times):
            if v > 0:
                ax_time.text(
                    bar.get_width() + ax_time.get_xlim()[1] * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}s", va="center", ha="left",
                    fontsize=7, color=TEXT_COLOR, alpha=0.75,
                )

        fig.suptitle(
            f"Benchmark: {scenario_name}",
            fontsize=14, fontweight="bold", color="#f0f6fc", y=0.995,
        )
        fig.text(
            0.5, 0.002,
            f"family={family}  •  5-fold CV  •  {n_contenders} contenders",
            ha="center", fontsize=8, color=TEXT_COLOR, alpha=0.6,
        )

        out_path = out_dir / f"{scenario_name}.png"
        fig.savefig(
            out_path, dpi=180, facecolor=BG_DARK, edgecolor="none",
            bbox_inches="tight", pad_inches=0.4,
        )
        plt.close(fig)
        paths.append(out_path)

    print(f"Generated {len(paths)} scenario figure(s) in {out_dir}/")
    return paths


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <results.json> <fig_dir> <figures.zip>")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    fig_dir = Path(sys.argv[2])
    zip_path = Path(sys.argv[3])

    payload = json.loads(results_path.read_text())
    results = payload.get("results", [])
    if not results:
        print("No results to plot.")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = generate_scenario_figures(results, fig_dir)

    if paths:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in paths:
                zf.write(p, arcname=p.name)
        print(f"Zipped {len(paths)} figure(s) -> {zip_path}")
    else:
        print("No figures generated.")


if __name__ == "__main__":
    main()
