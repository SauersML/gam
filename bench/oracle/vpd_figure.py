"""Figure of the vpd4l oracle's held-out comparison (#2951): per question, the log-score gain over the
`nothing` condition (nats per question, paired on the same questions, with +-1 standard error) of
reading the subcomponent's weights with its graph neighbourhood, its weights alone, and its activity on
other texts, on subcomponents of layers never trained on, on held-out texts (vpd_oracle.py compare's
summary).

  vpd_figure.py --summary SUMMARY.json --out FIGURE.png [--split heldout_layers]
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

QUESTIONS = {"activity": "activity\nlevel", "direction": "token up\nor down", "top": "most raised\ntoken", "which_upstream": "which input\ndrives it", "edge_cut": "cut edge:\ntoken up/down"}
ARMS = (("graph", "weights + graph", "#1f5fa8"), ("weights", "weights", "#6aa0d8"), ("activity", "activity on other texts", "#c0504d"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="heldout_layers")
    args = ap.parse_args()
    table = json.load(open(args.summary))
    questions = [q for q in QUESTIONS if any(f"{a}/{args.split}/{q}" in table for a, _, _ in ARMS)]
    fig, ax = plt.subplots(figsize=(2.4 * len(questions) + 2, 6))
    width = 0.8 / len(ARMS)
    x = np.arange(len(questions))
    for i, (arm, label, color) in enumerate(ARMS):
        rows = [table.get(f"{arm}/{args.split}/{q}") for q in questions]
        means = [r["gain_over_nothing_nats"] if r else np.nan for r in rows]
        errs = [r["standard_error_nats"] or 0 if r else 0 for r in rows]
        ax.bar(x + (i - (len(ARMS) - 1) / 2) * width, means, width, yerr=errs, color=color, capsize=4, label=label)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, [QUESTIONS[q] for q in questions], fontsize=15)
    ax.set_ylabel("log-score gain over no input\n(nats per question)", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=14, frameon=False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(args.out)


if __name__ == "__main__":
    main()
