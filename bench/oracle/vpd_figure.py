"""Figure of the vpd4l oracle's held-out comparison (#2951), from each condition's evaluation file: the
log-score gain over the `nothing` condition (nats per question, paired on the same questions, +-1
standard error) of reading a subcomponent's weights with its measured neighbourhood, its weights alone,
and its activity on other texts. Rows: subcomponents of the layer never trained on, and of trained
layers, both on held-out texts. Left: each question kind on the natural distribution. Right: the effect
questions (token direction and most raised token, pooled) by effect stratum, the decade of the removal KL.

With --previous (an earlier round's runs, each gain over that round's own `nothing`), the earlier
round's arms are drawn beside, hatched.

  vpd_figure.py --runs DIR... [--previous DIR...] --eval eval_labels_heldout.jsonl --out FIGURE.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

QUESTIONS = {"activity": "activity\nlevel", "direction": "token up\nor down", "top": "most raised\ntoken", "continuation": "amplified\ncontinuation",
             "edge": "cut edge", "attribution": "which raises\nthe prediction"}
STRATA = ("<1e-5", "1e-5", "1e-4", "1e-3", "1e-2", ">1e-1")
ARMS = (("graph_lens", "weights + neighbours + lens", "#0b3d91"), ("weights_lens", "weights + lens", "#2e86de"), ("graph", "weights + neighbours", "#1f5fa8"),
        ("weights", "weights", "#6aa0d8"), ("activity", "activity on other texts", "#c0504d"))
SPLITS = (("heldout_layers", "layer never trained on"), ("trained_layers", "trained layers"))


def gain(rows, base, keep) -> tuple[float, float]:
    d = np.array([r["log_score"] - b["log_score"] for r, b in zip(rows, base) if keep(r)])
    if len(d) < 2:
        return float("nan"), 0.0
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--previous", nargs="*", default=[])
    args = ap.parse_args()

    def load(dirs):
        out = {}
        for d in dirs:
            config = json.loads((Path(d) / "config.json").read_text())
            out[config["condition"]] = [json.loads(line) for line in open(Path(d) / args.eval)]
        return out

    sets = [("round 3: " if args.previous else "", load(args.runs), False)] + ([("round 2: ", load(args.previous), True)] if args.previous else [])
    # arms: (rows, base rows, label, color, hatched)
    arms = [(runs[a], runs["nothing"], prefix + label, color, old) for prefix, runs, old in sets for a, label, color in ARMS if a in runs]
    questions = [q for q in QUESTIONS if any(r["question"] == q for r in sets[0][1]["nothing"])]
    fig, axes = plt.subplots(2, 2, figsize=(20, 11), gridspec_kw={"width_ratios": [len(questions), len(STRATA)]})
    width = 0.8 / len(arms)
    for row, (split, split_label) in enumerate(SPLITS):
        for col in range(2):
            ax = axes[row, col]
            groups = questions if col == 0 else list(range(len(STRATA)))
            x = np.arange(len(groups))
            for i, (rows, base, label, color, old) in enumerate(arms):
                stats = []
                for g in groups:
                    if col == 0:
                        keep = lambda r, g=g: r["split"] == split and r["question"] == g and r["distribution"] == "natural"  # noqa: E731
                    else:
                        keep = lambda r, g=g: r["split"] == split and r["question"] in ("direction", "top") and r["distribution"] == "stratified" and r["stratum"] == g  # noqa: E731
                    stats.append(gain(rows, base, keep))
                ax.bar(x + (i - (len(arms) - 1) / 2) * width, [m for m, _ in stats], width, yerr=[e for _, e in stats], color="white" if old else color,
                       edgecolor=color, hatch="//" if old else None, capsize=3, label=label)
            ax.axhline(0, color="black", linewidth=1)
            ax.set_xticks(x, [QUESTIONS[q] for q in groups] if col == 0 else [f"KL {s}" for s in STRATA], fontsize=13)
            ax.tick_params(axis="y", labelsize=13)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if col == 0:
                ax.set_ylabel(f"{split_label}\ngain over no input (nats per question)", fontsize=14)
            else:
                ax.set_xlabel("token questions, by the removal's KL at the peak (nats)", fontsize=14)
    axes[0, 0].legend(fontsize=13, frameon=False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, facecolor="white")
    print(args.out)


if __name__ == "__main__":
    main()
