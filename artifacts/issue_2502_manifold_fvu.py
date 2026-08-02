#!/usr/bin/env python3
"""Render the fixed-bit manifold-SAE benchmark figure from its JSON artifact."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#0072B2"
VERMILION = "#D55E00"
GREEN = "#009E73"
GREY = "#8A8A8A"
INK = "#2F2F2F"
MUTED = "#707070"


def best_under(points, arms, bits_per_site):
    """Best point or time-sharing interpolation under a bit-budget cap."""
    candidates = []
    for arm in arms:
        xs = np.array([row[f"bits_{arm}"] for row in points])
        ys = np.array([row[f"fvu_{arm}"] for row in points])
        candidates.extend(ys[xs <= bits_per_site])
        if xs.min() <= bits_per_site <= xs.max():
            candidates.append(np.interp(bits_per_site, xs, ys))
    if not candidates:
        raise ValueError(f"no code fits under {bits_per_site} bits/site")
    return float(min(candidates))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    data = json.loads(args.data.read_text())
    layers = sorted((int(layer), result) for layer, result in data["layers"].items())
    budgets = data["protocol"]["total_bit_budgets"]

    # Aggregate rate-distortion points at each shared support size.
    topks = [row["topk"] for row in layers[0][1]["points"]]
    curves = {}
    for arm in ("lin", "man"):
        curves[arm] = {
            "bits": [
                sum(next(row[f"bits_{arm}"] for row in result["points"] if row["topk"] == k)
                    for _, result in layers)
                for k in topks
            ],
            "fvu": [
                np.mean([next(row[f"fvu_{arm}"] for row in result["points"] if row["topk"] == k)
                         for _, result in layers])
                for k in topks
            ],
        }

    fixed = {"linear": [], "enabled": []}
    for budget in budgets:
        fixed["linear"].append(np.mean([
            best_under(result["points"], ("lin",), budget / len(layers))
            for _, result in layers
        ]))
        fixed["enabled"].append(np.mean([
            best_under(result["points"], ("lin", "man"), budget / len(layers))
            for _, result in layers
        ]))

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.0), dpi=180)
    fig.patch.set_facecolor("#FCFCFB")
    for ax in axes:
        ax.set_facecolor("#FCFCFB")
        ax.grid(alpha=0.18, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)

    # A — the benchmark's actual currency: mean FVU against aggregate bits.
    ax = axes[0]
    ax.plot(curves["lin"]["bits"], curves["lin"]["fvu"], "o-", color=VERMILION,
            linewidth=2.0, markersize=6, label="linear amplitudes")
    ax.plot(curves["man"]["bits"], curves["man"]["fvu"], "s--", color=BLUE,
            linewidth=1.5, markersize=5, alpha=0.7, label="raw pair-chart candidates")
    envelope_bits = np.linspace(96, 576, 160)
    envelope_fvu = [
        np.mean([
            best_under(result["points"], ("lin", "man"), bits / len(layers))
            for _, result in layers
        ])
        for bits in envelope_bits
    ]
    ax.plot(envelope_bits, envelope_fvu, color=GREEN, linewidth=2.4,
            label="manifold-enabled envelope")
    for budget in budgets:
        ax.axvline(budget, color=GREY, linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_xlabel("total bits/token across four sites", color=INK, fontsize=9)
    ax.set_ylabel("mean held-out FVU (lower is better)", color=INK, fontsize=9)
    ax.set_title("Rate-distortion frontier", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    # B — a nested code family: charts are optional, so enabling them cannot
    # remove the linear baseline from the budget-feasible set.
    ax = axes[1]
    x = np.arange(len(budgets))
    width = 0.36
    ax.bar(x - width / 2, fixed["linear"], width, color=VERMILION, label="linear only")
    ax.bar(x + width / 2, fixed["enabled"], width, color=GREEN, label="manifold enabled")
    for i, (linear, enabled) in enumerate(zip(fixed["linear"], fixed["enabled"])):
        delta = enabled - linear
        ax.text(i, max(linear, enabled) + 0.008, f"Δ {delta:+.4f}", ha="center",
                va="bottom", color=GREEN if delta < 0 else MUTED, fontsize=8)
    ax.set_xticks(x, [str(budget) for budget in budgets])
    ax.set_xlabel("fixed total bits/token", color=INK, fontsize=9)
    ax.set_ylabel("mean held-out FVU", color=INK, fontsize=9)
    ax.set_title("Optional charts: no regression under a cap", color=INK, fontsize=11)
    ax.set_ylim(0, max(fixed["linear"] + fixed["enabled"]) * 1.12)
    ax.legend(frameon=False, fontsize=8)

    # C — why pair decoding was required, and proof it routes on held-out rows.
    ax = axes[2]
    labels = [f"L{layer}" for layer, _ in layers]
    pair = [result["census"]["pair_accepted"] for _, result in layers]
    single = [result["census"]["accepted"] - result["census"]["pair_accepted"]
              for _, result in layers]
    eval_rows = data["protocol"]["eval_rows_per_layer"]
    hit_rate = [
        100 * next(row["pair_hits"] for row in result["points"] if row["topk"] == 4) / eval_rows
        for _, result in layers
    ]
    x = np.arange(len(layers))
    ax.bar(x, pair, color=BLUE, label="accepted pair charts")
    ax.bar(x, single, bottom=pair, color=GREY, label="accepted single charts")
    ax.set_xticks(x, labels)
    ax.set_ylabel("accepted charts", color=INK, fontsize=9)
    ax.set_title("The accepted geometry is paired—and routes", color=INK, fontsize=11)
    ax2 = ax.twinx()
    ax2.plot(x, hit_rate, "D-", color=GREEN, linewidth=1.8, markersize=5,
             label="held-out rows using a pair chart (top-k 4)")
    ax2.set_ylabel("held-out pair-chart use (%)", color=GREEN, fontsize=9)
    ax2.tick_params(axis="y", colors=GREEN, labelsize=8)
    ax2.spines["top"].set_visible(False)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=7,
              loc="upper left")

    fig.suptitle(
        "Manifold SAE on GPT-2 small: pair charts improve FVU at fixed rate",
        color=INK,
        fontsize=13,
        y=1.01,
    )
    fig.text(
        0.5,
        0.012,
        "Held-out raw-space FVU, layers 3/5/7/9. G=1024 clears the census occupancy floor "
        "(1.35×); the discarded G=4096 arm was below it (0.42×).",
        ha="center",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(args.output, bbox_inches="tight", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
