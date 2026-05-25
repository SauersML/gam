from __future__ import annotations

import math
import typing
from pathlib import Path

import numpy as np
import pandas as pd


def configure(context: dict[str, typing.Any]) -> None:
    globals().update(context)


def _metric_display_config(family: str) -> typing.Any:
    """Return (metric_key, display_label, higher_is_better) tuples for a family."""
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
    # gaussian
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


def _style_plot_ax(ax: typing.Any, *, bg_card: str, text_color: str, grid_color: str) -> None:
    ax.set_facecolor(bg_card)
    ax.tick_params(colors=text_color, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(grid_color)
    ax.spines["left"].set_color(grid_color)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    ax.title.set_color(text_color)
    ax.grid(color=grid_color, linewidth=0.5, alpha=0.8, zorder=0)


def _subsample_plot_df(df: pd.DataFrame, max_points: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(df), size=max_points, replace=False))
    return typing.cast(pd.DataFrame, df.iloc[keep].copy())


def _plot_single_predictor_gaussian(ax: typing.Any, payload: dict[str, typing.Any], *, accent: str, text_color: str) -> None:
    df = pd.DataFrame(
        {
            "x": payload["x"],
            "actual": payload["actual"],
            "predicted": payload["predicted"],
        }
    ).sort_values("x")
    pts = _subsample_plot_df(df, max_points=3000)
    ax.scatter(pts["x"], pts["actual"], s=11, color="#8b949e", alpha=0.35, linewidths=0, zorder=2)
    ax.plot(df["x"], df["predicted"], color=accent, linewidth=2.2, alpha=0.95, zorder=3)
    ax.set_xlabel(str(payload["primary_feature"]))
    ax.set_ylabel(str(payload["target"]))
    ax.text(0.02, 0.96, "actual points + OOF prediction", transform=ax.transAxes, va="top", color=text_color, fontsize=8, alpha=0.75)


def _plot_single_predictor_binomial(ax: typing.Any, payload: dict[str, typing.Any], *, accent: str, text_color: str) -> None:
    df = pd.DataFrame(
        {
            "x": payload["x"],
            "actual": payload["actual"],
            "predicted": payload["predicted"],
        }
    ).sort_values("x")
    rng = np.random.default_rng(42)
    pts = _subsample_plot_df(df, max_points=2500)
    jitter = rng.normal(loc=0.0, scale=0.04, size=len(pts))
    ax.scatter(pts["x"], np.clip(pts["actual"] + jitter, -0.08, 1.08), s=12, color="#8b949e", alpha=0.35, linewidths=0, zorder=2)
    ax.plot(df["x"], np.clip(df["predicted"], 0.0, 1.0), color=accent, linewidth=2.2, alpha=0.95, zorder=3)
    bins = np.linspace(float(df["x"].min()), float(df["x"].max()), 20)
    if np.unique(bins).size >= 3:
        centers = 0.5 * (bins[:-1] + bins[1:])
        obs = []
        keep = []
        x_vals = df["x"].to_numpy(dtype=float)
        y_vals = df["actual"].to_numpy(dtype=float)
        for i in range(len(bins) - 1):
            mask = (x_vals >= bins[i]) & (x_vals < bins[i + 1] if i < len(bins) - 2 else x_vals <= bins[i + 1])
            if int(np.sum(mask)) >= 5:
                obs.append(float(np.mean(y_vals[mask])))
                keep.append(i)
        if keep:
            ax.plot(centers[keep], obs, color="#f2cc60", linewidth=1.6, linestyle="--", alpha=0.95, zorder=4)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel(str(payload["primary_feature"]))
    ax.set_ylabel(f"P({payload['target']}=1)")
    ax.text(0.02, 0.96, "jittered class points + OOF probability", transform=ax.transAxes, va="top", color=text_color, fontsize=8, alpha=0.75)


def _plot_actual_vs_predicted(ax: typing.Any, payload: dict[str, typing.Any], *, accent: str, text_color: str) -> None:
    df = pd.DataFrame({"actual": payload["actual"], "predicted": payload["predicted"]})
    pts = _subsample_plot_df(df, max_points=3000)
    ax.scatter(pts["actual"], pts["predicted"], s=14, color=accent, alpha=0.5, linewidths=0, zorder=3)
    lo = float(min(df["actual"].min(), df["predicted"].min()))
    hi = float(max(df["actual"].max(), df["predicted"].max()))
    ax.plot([lo, hi], [lo, hi], color="#8b949e", linewidth=1.2, linestyle="--", alpha=0.9, zorder=2)
    ax.set_xlabel(f"actual {payload['target']}")
    ax.set_ylabel(f"predicted {payload['target']}")
    ax.text(0.02, 0.96, "OOF predictions", transform=ax.transAxes, va="top", color=text_color, fontsize=8, alpha=0.75)


def _plot_binary_probability(ax: typing.Any, payload: dict[str, typing.Any], *, accent: str, text_color: str) -> None:
    df = pd.DataFrame({"actual": payload["actual"], "predicted": payload["predicted"]}).sort_values("predicted")
    pts = _subsample_plot_df(df, max_points=2500)
    rng = np.random.default_rng(42)
    jitter = rng.normal(loc=0.0, scale=0.04, size=len(pts))
    ax.scatter(np.clip(pts["predicted"], 0.0, 1.0), np.clip(pts["actual"] + jitter, -0.08, 1.08), s=12, color="#8b949e", alpha=0.35, linewidths=0, zorder=2)
    bins = np.linspace(0.0, 1.0, 16)
    centers = 0.5 * (bins[:-1] + bins[1:])
    obs = []
    keep = []
    p = np.clip(df["predicted"].to_numpy(dtype=float), 0.0, 1.0)
    y = df["actual"].to_numpy(dtype=float)
    for i in range(len(bins) - 1):
        mask = (p >= bins[i]) & (p < bins[i + 1] if i < len(bins) - 2 else p <= bins[i + 1])
        if int(np.sum(mask)) >= 5:
            obs.append(float(np.mean(y[mask])))
            keep.append(i)
    if keep:
        ax.plot(centers[keep], obs, color=accent, linewidth=2.2, zorder=3)
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#8b949e", linewidth=1.1, linestyle="--", alpha=0.8, zorder=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel(f"observed {payload['target']}")
    ax.text(0.02, 0.96, "jittered datapoints + binned event rate", transform=ax.transAxes, va="top", color=text_color, fontsize=8, alpha=0.75)


def _plot_survival_panel(ax: typing.Any, payload: dict[str, typing.Any], *, text_color: str) -> None:
    df = pd.DataFrame(
        {
            "time": payload["time"],
            "event": payload["event"],
            "risk": payload["risk"],
        }
    )
    if payload.get("pred_time") is not None:
        df["pred_time"] = payload["pred_time"]
    risk = df["risk"].to_numpy(dtype=float)
    q = np.unique(np.quantile(risk, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]))
    if q.size < 4:
        q = np.unique(np.quantile(risk, [0.0, 0.5, 1.0]))
    if q.size < 3:
        q = np.array([float(np.min(risk)), float(np.max(risk))], dtype=float)
    if q.size >= 3:
        group = pd.qcut(df["risk"], q=min(q.size - 1, 3), labels=False, duplicates="drop")
    else:
        group = pd.Series(np.zeros(len(df), dtype=int))
    df["group"] = group.fillna(0).astype(int)
    labels = ["Low risk", "Mid risk", "High risk"]
    colors = ["#56d364", "#d29922", "#f85149"]
    KaplanMeierFitter = _require_lifelines_kaplan_meier()
    kmf = KaplanMeierFitter()
    for grp in sorted(df["group"].unique()):
        sub = df[df["group"] == grp]
        if sub.empty:
            continue
        label = labels[min(int(grp), len(labels) - 1)]
        color = colors[min(int(grp), len(colors) - 1)]
        kmf.fit(sub["time"], event_observed=sub["event"], label=f"{label} (n={len(sub)})")
        sf = kmf.survival_function_
        ax.step(sf.index.to_numpy(dtype=float), sf.iloc[:, 0].to_numpy(dtype=float), where="post", color=color, linewidth=2.0, alpha=0.95, zorder=3)
        censor_times = sub.loc[sub["event"] <= 0.5, "time"].to_numpy(dtype=float)
        if censor_times.size:
            censor_y = kmf.predict(censor_times)
            ax.scatter(censor_times, np.asarray(censor_y, dtype=float), marker="|", s=36, color=color, alpha=0.9, linewidths=1.1, zorder=4)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(str(payload["time_col"]))
    ax.set_ylabel("survival probability")
    event_rate = float(np.mean(df["event"].to_numpy(dtype=float) > 0.5))
    summary = f"risk-stratified KM • event rate {event_rate:.1%}"
    if "pred_time" in df.columns:
        summary += f" • median predicted time {float(np.median(df['pred_time'])):.2f}"
    ax.text(0.02, 0.96, summary, transform=ax.transAxes, va="top", color=text_color, fontsize=8, alpha=0.75)


def generate_scenario_datapoint_figures(results: list[dict[str, typing.Any]], out_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available — skipping datapoint figure generation.")
        return []

    from collections import defaultdict

    by_scenario: dict[str, list[dict[str, typing.Any]]] = defaultdict(list)
    for r in results:
        if r.get("status") == "ok" and r.get("plot_payload"):
            by_scenario[r["scenario_name"]].append(r)

    bg_dark = "#0d1117"
    bg_card = "#161b22"
    text_color = "#c9d1d9"
    grid_color = "#21262d"
    accent_colors = [
        "#58a6ff", "#3fb950", "#d29922", "#f85149",
        "#bc8cff", "#79c0ff", "#56d364", "#e3b341",
    ]
    paths: list[Path] = []
    for scenario_name, rows in sorted(by_scenario.items()):
        if not rows:
            continue
        n_panels = len(rows)
        ncols = 2 if n_panels > 1 else 1
        nrows = int(math.ceil(n_panels / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(14, max(4.5, 4.2 * nrows)),
            facecolor=bg_dark,
            squeeze=False,
        )
        axes_flat = list(axes.reshape(-1))
        for ax in axes_flat:
            _style_plot_ax(ax, bg_card=bg_card, text_color=text_color, grid_color=grid_color)
        for idx, row in enumerate(rows):
            ax = axes_flat[idx]
            payload = row["plot_payload"]
            accent = accent_colors[idx % len(accent_colors)]
            family = str(payload["family"])
            if family == "survival":
                _plot_survival_panel(ax, payload, text_color=text_color)
            elif payload.get("primary_feature"):
                if family == "gaussian":
                    _plot_single_predictor_gaussian(ax, payload, accent=accent, text_color=text_color)
                else:
                    _plot_single_predictor_binomial(ax, payload, accent=accent, text_color=text_color)
            elif family == "gaussian":
                _plot_actual_vs_predicted(ax, payload, accent=accent, text_color=text_color)
            else:
                _plot_binary_probability(ax, payload, accent=accent, text_color=text_color)
            ax.set_title(_short_contender_label(str(row["contender"])), fontsize=10, loc="left", pad=8)
        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)
        fig.suptitle(f"Benchmark Datapoints: {scenario_name}", fontsize=14, fontweight="bold", color="#f0f6fc", y=0.995)
        fig.text(0.5, 0.002, "Out-of-fold predictions with raw observed datapoints", ha="center", fontsize=8, color=text_color, alpha=0.65)
        out_path = out_dir / f"{scenario_name}__datapoints.png"
        fig.savefig(out_path, dpi=180, facecolor=bg_dark, edgecolor="none", bbox_inches="tight", pad_inches=0.35)
        plt.close(fig)
        paths.append(out_path)
    print(f"Generated {len(paths)} datapoint figure(s) in {out_dir}/")
    return paths


def generate_scenario_figures(results: list[dict[str, typing.Any]], out_dir: Path) -> list[Path]:
    """Create one beautiful comparison PNG per scenario."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available — skipping figure generation.")
        return []

    # Group results by scenario.
    from collections import defaultdict
    by_scenario: dict[str, list[dict[str, typing.Any]]] = defaultdict(list)
    for r in results:
        if r.get("status") == "ok":
            by_scenario[r["scenario_name"]].append(r)

    # ---- Theme ----
    bg_dark = "#0d1117"
    bg_card = "#161b22"
    text_color = "#c9d1d9"
    grid_color = "#21262d"
    accent_colors = [
        "#58a6ff", "#3fb950", "#d29922", "#f85149",
        "#bc8cff", "#79c0ff", "#56d364", "#e3b341",
        "#ff7b72", "#d2a8ff", "#a5d6ff", "#7ee787",
        "#f2cc60", "#ffa198", "#cabffd", "#b1bac4",
    ]

    paths = []
    for scenario_name, rows in sorted(by_scenario.items()):
        if not rows:
            continue
        family = rows[0].get("family", "gaussian")
        evaluation = str(rows[0].get("evaluation", "unknown"))
        metrics_cfg = _metric_display_config(family)
        # Filter to metrics that have at least one non-None value.
        active_metrics = []
        for key, label, hib in metrics_cfg:
            metric_values = [r.get(key) for r in rows if r.get(key) is not None]
            if metric_values:
                active_metrics.append((key, label, hib))
        if not active_metrics:
            continue

        n_metrics = len(active_metrics)
        contenders = [r["contender"] for r in rows]
        n_contenders = len(contenders)

        fig_height = max(3.2, 1.2 + 0.42 * n_contenders) * (n_metrics + 1)
        fig, axes = plt.subplots(
            n_metrics + 1, 1,
            figsize=(10, min(fig_height, 28)),
            facecolor=bg_dark,
            gridspec_kw={"hspace": 0.45},
        )
        if n_metrics + 1 == 1:
            axes = [axes]

        for ax in axes:
            ax.set_facecolor(bg_card)
            ax.tick_params(colors=text_color, which="both")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color(grid_color)
            ax.spines["left"].set_color(grid_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)

        short_labels = [_short_contender_label(c) for c in contenders]
        y_pos = list(range(n_contenders))
        colors = [accent_colors[i % len(accent_colors)] for i in range(n_contenders)]

        for idx, (key, label, higher_is_better) in enumerate(active_metrics):
            ax = axes[idx]
            metric_vals: list[float] = []
            for r in rows:
                v = r.get(key)
                metric_vals.append(float(v) if v is not None else float("nan"))

            bars = ax.barh(
                y_pos, metric_vals, height=0.62,
                color=colors, edgecolor="none", alpha=0.88,
                zorder=3,
            )
            # Highlight the best value.
            valid_vals = [v for v in metric_vals if np.isfinite(v)]
            if valid_vals:
                if higher_is_better:
                    best_val = max(valid_vals)
                else:
                    best_val = min(valid_vals)
                for i, (bar, v) in enumerate(zip(bars, metric_vals)):
                    if np.isfinite(v) and abs(v - best_val) < 1e-12:
                        bar.set_edgecolor("#f0f6fc")
                        bar.set_linewidth(1.8)
                        bar.set_alpha(1.0)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(short_labels, fontsize=8.5)
            ax.invert_yaxis()
            ax.set_xlabel(label, fontsize=9.5, fontweight="bold")
            direction_hint = "↑ higher is better" if higher_is_better else "↓ lower is better"
            ax.set_title(f"{label}  ({direction_hint})", fontsize=10, loc="left", pad=8)
            ax.grid(axis="x", color=grid_color, linewidth=0.5, zorder=0)

            # Value labels on bars.
            for bar, v in zip(bars, metric_vals):
                if not np.isfinite(v):
                    continue
                if abs(v) < 0.01:
                    fmt = f"{v:.4f}"
                elif abs(v) < 1.0:
                    fmt = f"{v:.4f}"
                elif abs(v) < 100:
                    fmt = f"{v:.3f}"
                else:
                    fmt = f"{v:.1f}"
                ax.text(
                    bar.get_width() + ax.get_xlim()[1] * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    fmt, va="center", ha="left",
                    fontsize=7.5, color=text_color, alpha=0.85,
                )

        # Timing subplot: fit + predict seconds.
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
        ax_time.set_xlabel("Time (seconds)", fontsize=9.5, fontweight="bold")
        ax_time.set_title("Fit + Predict Time  (↓ lower is better)", fontsize=10, loc="left", pad=8)
        ax_time.grid(axis="x", color=grid_color, linewidth=0.5, zorder=0)
        ax_time.legend(
            loc="lower right", fontsize=8,
            facecolor=bg_card, edgecolor=grid_color,
            labelcolor=text_color,
        )

        for bar, v in zip(bars_fit, fit_times):
            if v > 0:
                ax_time.text(
                    bar.get_width() + ax_time.get_xlim()[1] * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}s", va="center", ha="left",
                    fontsize=7, color=text_color, alpha=0.75,
                )
        for bar, v in zip(bars_pred, pred_times):
            if v > 0:
                ax_time.text(
                    bar.get_width() + ax_time.get_xlim()[1] * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}s", va="center", ha="left",
                    fontsize=7, color=text_color, alpha=0.75,
                )

        # Suptitle.
        fig.suptitle(
            f"Benchmark: {scenario_name}",
            fontsize=14, fontweight="bold",
            color="#f0f6fc", y=0.995,
        )
        fig.text(
            0.5, 0.002,
            f"family={family}  •  {evaluation}  •  {n_contenders} contenders",
            ha="center", fontsize=8, color=text_color, alpha=0.6,
        )

        out_path = out_dir / f"{scenario_name}.png"
        fig.savefig(
            out_path, dpi=180,
            facecolor=bg_dark, edgecolor="none",
            bbox_inches="tight", pad_inches=0.4,
        )
        plt.close(fig)
        paths.append(out_path)

    print(f"Generated {len(paths)} scenario figure(s) in {out_dir}/")
    return paths


def zip_figure_dir(fig_dir: Path, zip_path: Path) -> None:
    """Bundle all PNGs in fig_dir into a single .zip for GHA artifact download."""
    import zipfile
    pngs = sorted(fig_dir.glob("*.png"))
    if not pngs:
        print("No figures to zip.")
        return
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in pngs:
            zf.write(p, arcname=p.name)
    print(f"Zipped {len(pngs)} figure(s) -> {zip_path}")



