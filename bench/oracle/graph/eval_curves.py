"""Held-out curves (#2951 graph oracle): per question, the KL of the model's prediction from the graph (bits, run alone
over changed prompts) against description length, for the oracle's best answer, the search's answer, VPD's answer and
the empty graph; left, a few questions' curves; right, each method's curve area per question (score.key) against the
search's.

  eval_curves.py EVAL_SAMPLES_JSONL OUT.png [--step N] [--examples 4]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402

COLORS = {"oracle": "#1f4e79", "search": "#7f7f7f", "vpd": "#b03a2e"}
NAMES = {"oracle": "oracle", "search": "search", "vpd": "VPD's answer"}


def running(curve, hi):
    """The curve as a step function: (bits, lowest KL so far) points, extended to hi."""
    pts, best = [], math.inf
    for b, k in sorted(curve):
        best = min(best, k)
        pts.append((max(b, 1.0), best))
    pts.append((hi, best))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("out")
    ap.add_argument("--step", type=int)
    ap.add_argument("--examples", type=int, default=4)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.samples)]
    step = a.step if a.step is not None else max(r["step"] for r in rows)
    by = {}
    for r in rows:
        if r["step"] != step or not r["score"].get("valid", True) or not r["score"].get("curve"):
            continue
        q = by.setdefault(r["behavior"], {})
        name = r["program"]
        if name == "oracle":
            if "oracle" not in q or score.key(r["score"]) < score.key(q["oracle"]):
                q["oracle"] = r["score"]
        elif name in ("search", "vpd", "empty"):
            q[name] = r["score"]
    qs = [b for b, q in sorted(by.items()) if "oracle" in q and "search" in q]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for i, b in enumerate(qs[:a.examples]):
        q = by[b]
        for name in ("search", "vpd", "oracle"):
            if name in q:
                pts = running(q[name]["curve"], q[name]["hi"])
                ax.step([p[0] for p in pts], [p[1] for p in pts], where="post", color=COLORS[name], lw=2, alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("description length (bits)", fontsize=15)
    ax.set_ylabel("KL of the prediction (bits)", fontsize=15)
    for name in ("oracle", "search", "vpd"):
        ax.plot([], [], color=COLORS[name], lw=2, label=NAMES[name])
    ax.legend(fontsize=13, frameon=False)
    ax = axes[1]
    xs = [score.key(by[b]["search"])[1] for b in qs]
    for name in ("oracle", "vpd"):
        ys = [(score.key(by[b]["search"])[1], score.key(by[b][name])[1]) for b in qs if name in by[b]]
        if ys:
            ax.scatter([y[0] for y in ys], [y[1] for y in ys], s=40, color=COLORS[name], alpha=0.8, label=NAMES[name])
    if xs:
        lim = [min(xs + [score.key(by[b]["oracle"])[1] for b in qs]) * 0.9, max(xs + [score.key(by[b]["oracle"])[1] for b in qs]) * 1.1]
        ax.plot(lim, lim, color="#7f7f7f", lw=1)
    ax.set_xlabel("search's curve area (bits)", fontsize=15)
    ax.set_ylabel("curve area (bits)", fontsize=15)
    ax.legend(fontsize=13, frameon=False)
    for ax in axes:
        ax.tick_params(labelsize=13)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130, facecolor="white")
    wins = sum(score.key(by[b]["oracle"])[1] < score.key(by[b]["search"])[1] for b in qs)
    print(f"{a.out}: {len(qs)} questions, oracle below search on {wins}")


if __name__ == "__main__":
    main()
