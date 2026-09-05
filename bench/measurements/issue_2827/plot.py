"""Separate interpolation setup from optimization in a spatial stage trace.

Usage: python plot.py INPUT.log OUTPUT.png OUTPUT.csv
"""

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    source, figure_path, table_path = map(Path, sys.argv[1:])
    lines = source.read_text().splitlines()
    node_values = []
    optimizer_values = []
    for line in lines:
        node = re.search(r'ensure_theta .*psi=\["([+-]?[0-9.eE]+)"\]', line)
        if node:
            node_values.append(float(node.group(1)))
        trial = re.search(r'\[KAPPA-PHASE\].*psi=\[([+0-9.eE-]+)\]', line)
        if trial:
            optimizer_values.append(float(trial.group(1)))
    # The source revision behind this trace fixes the interpolation at 513
    # first-kind nodes. The remaining realizations are checks and exact solves.
    node_count = 513
    measured = np.asarray(node_values[:node_count])
    if measured.size != node_count or not optimizer_values:
        raise ValueError("trace lacks the fixed-node build or optimizer phase")
    cosine = np.cos(np.pi * (2 * np.arange(node_count) + 1) / (2 * node_count))
    # Recover the two endpoint values lost to six-decimal stage formatting.
    center, halfwidth = np.linalg.lstsq(
        np.column_stack([np.ones(node_count), cosine]), measured, rcond=None
    )[0]
    predicted = center + halfwidth * cosine
    residual = measured - predicted
    with table_path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["phase", "evaluation", "psi", "cosine_prediction", "residual"])
        for i, (value, fitted, error) in enumerate(zip(measured, predicted, residual), 1):
            writer.writerow(["interpolation_setup", i, value, fitted, error])
        for i, value in enumerate(optimizer_values, 1):
            writer.writerow(["optimizer_callback", i, value, "", ""])
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    figure.suptitle("The 513-point trajectory is interpolation setup", fontsize=18, fontweight="bold")
    axes[0].plot(np.arange(1, node_count + 1), predicted, color="#264f9e", linewidth=2.5, label="Chebyshev cosine nodes")
    axes[0].scatter(np.arange(1, node_count + 1)[::12], measured[::12], s=20, color="#ed9b34", zorder=3, label="Logged design realizations")
    axes[0].set(xlabel="Setup design realization", ylabel="Spatial coordinate ψ")
    axes[0].legend(frameon=False)
    axes[0].text(0.02, 0.08, "Setup: 11 s → 4 min 48 s\nOptimizer has not started", transform=axes[0].transAxes, color="#264f9e")
    axes[1].plot(np.arange(1, len(optimizer_values) + 1), optimizer_values, marker="o", markersize=3.5, color="#177d60")
    axes[1].set(xlabel="Actual optimizer callback (including value probes and polish)", ylabel="Spatial coordinate ψ")
    axes[1].text(0.32, 0.78, "Actual seed: ψ = 0\nFirst trial reaches ψ = −0.511\nWinning BFGS solve: 15 iterations, 3.763 s", transform=axes[1].transAxes, color="#177d60")
    for axis in axes:
        axis.grid(alpha=0.16)
    figure.savefig(figure_path, dpi=160)
    print(json.dumps({
        "setup_nodes": node_count,
        "total_design_realizations": len(node_values),
        "optimizer_callbacks": len(optimizer_values),
        "recovered_window": [float(center - halfwidth), float(center + halfwidth)],
        "maximum_cosine_residual": float(np.abs(residual).max()),
        "actual_seed": optimizer_values[0],
        "first_optimizer_trial": optimizer_values[1],
    }, indent=2))


if __name__ == "__main__":
    main()
