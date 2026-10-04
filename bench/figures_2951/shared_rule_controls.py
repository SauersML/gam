"""Plot measured shared-rule controls; no model fitting or metric recomputation.

python shared_rule_controls.py --open
"""
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data/shared_rule_controls.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "shared_rule_controls")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 17, "axes.titlesize": 21, "axes.labelsize": 17,
                        "axes.spines.top": False, "axes.spines.right": False,
                        "axes.grid": False, "figure.facecolor": "white",
                        "axes.facecolor": "white", "pdf.fonttype": 42})
    labels = {"learned": "Learned shared body", "frozen_native": "Frozen native body",
              "frozen_random": "Frozen random body", "untied": "Separate bodies"}
    colors = {"learned": "#CB552F", "frozen_native": "#156BA3",
              "frozen_random": "#8D779C", "untied": "#6A7775"}
    order = list(labels)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for arm in order:
        h = data["initial"][arm]["history"]
        axes[0, 0].plot([r["step"] for r in h], [r["training_max"] for r in h],
                        color=colors[arm], linewidth=1.7, label=labels[arm])
    axes[0, 0].set(title="Initial fit: training error", xlabel="Optimizer updates",
                   ylabel="Maximum normalized output error", yscale="log")
    axes[0, 0].legend(frameon=False, fontsize=13)

    for i, arm in enumerate(order):
        values = data["initial"][arm]["saved_eval"]
        axes[0, 1].plot([i - .10, i + .10], values, color=colors[arm], linewidth=2)
        axes[0, 1].scatter([i - .10], [values[0]], color=colors[arm], marker="o", s=80)
        axes[0, 1].scatter([i + .10], [values[1]], color=colors[arm], marker="s", s=80)
        axes[0, 1].annotate(f"{values[1]:.2f}", (i + .10, values[1]),
                           xytext=(0, 9), textcoords="offset points", ha="center", fontsize=15)
    axes[0, 1].scatter([], [], color="black", marker="o", label="Layer 0")
    axes[0, 1].scatter([], [], color="black", marker="s", label="Layer 1")
    axes[0, 1].legend(frameon=False, fontsize=13)
    axes[0, 1].set(title="Initial fit: held-out error", ylabel="Maximum normalized output error",
                   xticks=range(4), xticklabels=["Learned", "Frozen\nnative", "Frozen\nrandom", "Separate"], ylim=(-.2, 7.4))

    for arm in ["learned", "frozen_native"]:
        h = data["warmstart"][arm]["history"]
        axes[1, 0].plot([r["step"] for r in h], [r["training_max"] for r in h],
                        color=colors[arm], linewidth=1.7, label=labels[arm])
    start = data["warmstart"]["learned"]["history"][0]["training_max"]
    axes[1, 0].axhline(start, color="#555555", linestyle=":", label="Common starting program")
    axes[1, 0].set(title="Same starting program: training error", xlabel="Additional optimizer updates",
                   ylabel="Maximum normalized output error", yscale="log")
    axes[1, 0].legend(frameon=False, fontsize=13)

    arms = ["learned", "frozen_native"]
    values = [data["warmstart"][arm]["saved_eval_max"] for arm in arms]
    axes[1, 1].bar(range(2), values, color=[colors[a] for a in arms], width=.48)
    for i, value in enumerate(values):
        axes[1, 1].text(i, value + .14, f"{value:.2f}", ha="center", fontsize=17)
    axes[1, 1].set(title="After warmstart: held-out error",
                   ylabel="Maximum normalized output error",
                   xticks=range(2), xticklabels=["Learned shared body", "Frozen native body"], ylim=(0, 7.4))
    fig.tight_layout(pad=2.2)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(args.out / f"controls.{suffix}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    if "larger64" in data:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
        for j, (key, label, color) in enumerate([
                ("warmstart", "8 training passages", "#899AA6"),
                ("larger64", "64 training passages", "#156BA3")]):
            xs = [i + (j - .5) * .30 for i in range(2)]
            for ax, metric in zip(axes, ["saved_eval_max", "optimizer_seconds"]):
                values = [data[key][arm][metric] for arm in arms]
                ax.bar(xs, values, width=.28, color=color, label=label)
                for x, value in zip(xs, values):
                    ax.annotate(f"{value:.2f}" if metric == "saved_eval_max" else f"{value:.1f}",
                                (x, value), xytext=(0, 6), textcoords="offset points",
                                ha="center", fontsize=16)
        for ax in axes:
            ax.set_xticks(range(2), ["Learned shared", "Frozen native"])
        axes[0].set(title="More fitting data improves held-out error",
                    ylabel="Maximum normalized output error", ylim=(0, 7.4))
        axes[1].set(title="Both fits use 256 additional updates",
                    ylabel="GPU fitting time (seconds)", ylim=(0, 220))
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=2, frameon=False, fontsize=16)
        fig.tight_layout(rect=(0, .11, 1, 1), pad=2)
        for suffix in ["pdf", "svg", "png"]:
            fig.savefig(args.out / f"data_coverage.{suffix}", bbox_inches="tight", dpi=150)
        plt.close(fig)
    if args.open:
        subprocess.run(["open", "-a", "Preview", str(args.out / "controls.pdf")], check=True)
        if "larger64" in data:
            subprocess.run(["open", "-a", "Preview", str(args.out / "data_coverage.pdf")], check=True)


if __name__ == "__main__":
    main()
