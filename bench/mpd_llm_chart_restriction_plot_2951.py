"""#2951 figure: next-token KL against the per-token rank of the MLP-input restriction, per layer."""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    base = json.load(open(sys.argv[1]))
    fisher = {}
    for path in sys.argv[2:-1]:
        fisher.update(json.load(open(path))["layers"])
    out = sys.argv[-1]
    layers = sorted(base["layers"], key=int)
    fig, axes = plt.subplots(1, len(layers), figsize=(5.2 * len(layers), 4.6), sharey=False)
    colors = {16: "#9ecae1", 128: "#4292c6", 1024: "#08306b"}
    for ax, layer in zip(axes, layers):
        e = base["layers"][layer]
        r = sorted(int(k) for k in e["global"])
        ax.plot(r, [e["global"][str(k)] for k in r], "o-", color="#636363", label="global subspace (PCA)")
        for C in (16, 128, 1024):
            pts = sorted((int(k.split("x")[1]), v) for k, v in e["atlas"].items() if int(k.split("x")[0]) == C)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=colors[C], label=f"atlas, {C} local charts")
        pts = sorted((int(k.split("x")[1]), v) for k, v in e["random"].items() if int(k.split("x")[0]) == 1024)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "x--", color="#e6550d", label="1024 random cells (control)")
        if layer in fisher and "fisher_global" in fisher[layer]:
            f = fisher[layer]
            r = sorted(int(k) for k in f["fisher_global"])
            ax.plot(r, [f["fisher_global"][str(k)] for k in r], "s-", color="#31a354", label="global, Fisher-optimal")
            pts = sorted((int(k.split("x")[1]), v) for k, v in f["fisher_atlas"].items())
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "s-", color="#006d2c", label="128 charts, Fisher-optimal")
        ax.axhline(e["zero_mean_input"], color="k", lw=0.8, ls=":", label="input replaced by its mean")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, None)
        ax.set_title(f"layer {layer} MLP input")
        ax.set_xlabel("rank kept per token")
    axes[0].set_ylabel("next-token KL to the model (nats)")
    axes[1].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
