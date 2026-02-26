#!/usr/bin/env python3
"""
Build a 1D ordering of subpopulation centroids in 16-PC space and plot:
1) PC1/PC2 centroid map color-coded by assigned prevalence
2) Full ordered list table on the same figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pairwise_dist(x: np.ndarray) -> np.ndarray:
    # Squared-distance expansion with clipping for numerical noise.
    s = np.sum(x * x, axis=1, keepdims=True)
    d2 = s + s.T - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def path_length(path: np.ndarray, d: np.ndarray) -> float:
    if len(path) <= 1:
        return 0.0
    return float(np.sum(d[path[:-1], path[1:]]))


def nearest_neighbor_path(d: np.ndarray, start: int) -> np.ndarray:
    n = d.shape[0]
    unused = set(range(n))
    unused.remove(start)
    path = [start]
    while unused:
        cur = path[-1]
        nxt = min(unused, key=lambda j: d[cur, j])
        path.append(nxt)
        unused.remove(nxt)
    return np.array(path, dtype=int)


def two_opt_open(path: np.ndarray, d: np.ndarray, max_passes: int = 20) -> np.ndarray:
    n = len(path)
    if n < 4:
        return path.copy()
    p = path.copy()
    for _ in range(max_passes):
        improved = False
        for i in range(n - 3):
            a, b = p[i], p[i + 1]
            for j in range(i + 2, n - 1):
                c, e = p[j], p[j + 1]
                before = d[a, b] + d[c, e]
                after = d[a, c] + d[b, e]
                if after + 1e-12 < before:
                    p[i + 1 : j + 1] = p[i + 1 : j + 1][::-1]
                    improved = True
        if not improved:
            break
    return p


def best_order(d: np.ndarray) -> np.ndarray:
    n = d.shape[0]
    best = None
    best_len = np.inf
    for s in range(n):
        p0 = nearest_neighbor_path(d, s)
        p1 = two_opt_open(p0, d)
        l = path_length(p1, d)
        if l < best_len:
            best_len = l
            best = p1
    assert best is not None
    return best


def draw_table_text(
    ax: plt.Axes, ordered_names: list[str], prevalence: np.ndarray, n_cols: int = 3
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
    x_positions = np.linspace(0.0, 0.68, len(col_blocks))
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
        default="bench/datasets/hgdp_1kg_pc_data.tsv",
        help="Input TSV with PC1..PC16 and Subpopulation",
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

    inp = Path(args.input)
    out_png = Path(args.out_png)
    out_csv = Path(args.out_csv)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, sep="\t")
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
    x = centroids[pc_cols].to_numpy(dtype=float)
    d = pairwise_dist(x)
    order_idx = best_order(d)

    ordered = centroids.iloc[order_idx].reset_index(drop=True)
    n = len(ordered)
    prevalence = np.linspace(0.02, 0.40, n)  # synthetic monotone low->high risk
    ordered["order"] = np.arange(1, n + 1)
    ordered["assigned_prevalence"] = prevalence
    ordered.to_csv(out_csv, index=False)

    fig = plt.figure(figsize=(24, 14), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35])
    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1])

    p1 = ordered["PC1"].to_numpy()
    p2 = ordered["PC2"].to_numpy()
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
        "Subpopulation Centroids in 16-PC Space\n1D order projected on PC1/PC2 (numbers = order)",
        fontsize=13,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25, linestyle="--")

    ordered_names = ordered["Subpopulation"].astype(str).tolist()
    draw_table_text(ax_tbl, ordered_names, prevalence, n_cols=3)

    fig.suptitle(
        "Centroid-based 1D Ordering of Subpopulations (16-PC Euclidean, NN + 2-opt path)",
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
