#!/usr/bin/env python3
"""
Plot an ordered view of subpopulation centroids in 16-PC space:
1) PC1/PC2 centroid map color-coded by assigned prevalence
2) Full ordered list table on the same figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd


def draw_table_text(
    ax: Axes, ordered_names: list[str], prevalence: list[float], n_cols: int = 3
) -> None:
    lines = [
        f"{i+1:>3}. {name:<30} prev={prevalence[i]:.3f}"
        for i, name in enumerate(ordered_names)
    ]
    n = len(lines)
    col_size = (n + n_cols - 1) // n_cols
    col_blocks = []
    for c in range(n_cols):
        start = c * col_size
        end = min((c + 1) * col_size, n)
        if start >= n:
            continue
        col_blocks.append("\n".join(lines[start:end]))

    ax.axis("off")
    ax.set_title("Full 1D Subpopulation Order", loc="left", fontsize=11, pad=6)
    if len(col_blocks) <= 1:
        x_positions = [0.0]
    else:
        step = 0.68 / (len(col_blocks) - 1)
        x_positions = [step * i for i in range(len(col_blocks))]
    for x, block in zip(x_positions, col_blocks):
        ax.text(
            x,
            1.0,
            block,
            va="top",
            ha="left",
            family="monospace",
            fontsize=7.3,
            transform=ax.transAxes,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=None,
        help="Optional TSV with PC1..PC16 and Subpopulation; defaults to the synthetic in-repo panel",
    )
    parser.add_argument(
        "--out-png",
        default="bench/artifacts/subpop_16pc_order.png",
        help="Output figure PNG",
    )
    parser.add_argument(
        "--out-csv",
        default="bench/artifacts/subpop_16pc_order.csv",
        help="Output ordered table CSV",
    )
    args = parser.parse_args()

    out_png = Path(args.out_png)
    out_csv = Path(args.out_csv)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.input:
        df = pd.read_csv(Path(args.input), sep="\t")
    else:
        from run_suite import _synthetic_hgdp_1kg_pc_panel

        df = _synthetic_hgdp_1kg_pc_panel()
    pc_cols = [f"PC{i}" for i in range(1, 17)]
    need = {"Subpopulation", "PC1", "PC2", *pc_cols}
    missing = need.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    centroids = (
        df.groupby("Subpopulation", dropna=False)[pc_cols]
        .mean(numeric_only=True)
        .dropna()
        .reset_index()
    )
    if "order" in df.columns:
        input_order = (
            df[["Subpopulation", "order"]]
            .dropna()
            .drop_duplicates(subset=["Subpopulation"], keep="first")
        )
        centroids = centroids.merge(input_order, on="Subpopulation", how="left")
        if centroids["order"].isna().any():
            missing_order = centroids.loc[
                centroids["order"].isna(), "Subpopulation"
            ].astype(str)
            raise RuntimeError(
                "Missing order for subpopulation(s): "
                + ", ".join(sorted(missing_order.tolist()))
            )
        centroids = centroids.sort_values(["order", "Subpopulation"], kind="stable")
    else:
        centroids = centroids.sort_values("Subpopulation", kind="stable")

    ordered = centroids.drop(columns=["order"], errors="ignore").reset_index(drop=True)
    n = len(ordered)
    prevalence = [
        0.02 if n <= 1 else 0.02 + (0.40 - 0.02) * i / (n - 1)
        for i in range(n)
    ]
    ordered["order"] = list(range(1, n + 1))
    ordered["assigned_prevalence"] = prevalence
    ordered.to_csv(out_csv, index=False)

    fig = plt.figure(figsize=(24, 14), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35])
    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1])

    p1 = ordered["PC1"].tolist()
    p2 = ordered["PC2"].tolist()
    sc = ax.scatter(
        p1,
        p2,
        c=prevalence,
        cmap="viridis",
        s=120,
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
    )
    ax.plot(p1, p2, color="#4a4a4a", linewidth=1.0, alpha=0.6, zorder=2)
    for i, row in ordered.iterrows():
        ax.text(
            row["PC1"],
            row["PC2"],
            str(i + 1),
            fontsize=7,
            ha="center",
            va="center",
            color="white",
            zorder=4,
        )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Assigned disease prevalence", fontsize=10)

    ax.set_title(
        "Subpopulation Centroids in 16-PC Space\nOrdered list projected on PC1/PC2 (numbers = order)",
        fontsize=13,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25, linestyle="--")

    ordered_names = ordered["Subpopulation"].astype(str).tolist()
    draw_table_text(ax_tbl, ordered_names, prevalence, n_cols=3)

    fig.suptitle(
        "Centroid-based Ordering of Subpopulations",
        fontsize=15,
        y=1.02,
    )
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_csv}")
    print(f"Subpopulations: {n}")


if __name__ == "__main__":
    main()
