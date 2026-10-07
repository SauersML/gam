"""Figures of an oracle training run (#2951): report.py RUN_DIR [--out PNG]

Left: S in bits per training step (the group mean and the mean of each group's best program) with the
share of valid programs. Right: the latest evaluation, per set (held-out behaviors, held-out prompts), the
oracle's mean single-sample S beside each baseline's S on the same behaviors and experiments. Prints the
numbers it plots as one JSON object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out")
    a = ap.parse_args()
    run = Path(a.run)
    steps = [json.loads(line) for line in open(run / "train.jsonl")] if (run / "train.jsonl").exists() else []
    evals = [json.loads(line) for line in open(run / "eval.jsonl")] if (run / "eval.jsonl").exists() else []
    summary = next((e for e in reversed(evals) if "summary" in e), None)
    plt.rcParams.update({"font.size": 15, "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 6))
    if steps:
        x = [s["step"] for s in steps]
        left.plot(x, [s["mean_bits"] for s in steps], color="#1f5fa8", lw=2)
        left.plot(x, [s["best_bits"] for s in steps], color="#c2410c", lw=2)
        left.text(x[-1], steps[-1]["mean_bits"], "  group mean", color="#1f5fa8", va="center")
        left.text(x[-1], steps[-1]["best_bits"], "  best of group", color="#c2410c", va="center")
        left.set_xlabel("training step")
        left.set_ylabel("S (bits)")
        twin = left.twinx()
        twin.plot(x, [s["valid_fraction"] for s in steps], color="#6b7280", lw=1.5, ls="--")
        twin.set_ylim(0, 1.05)
        twin.set_ylabel("valid share", color="#6b7280")
        for side in ("top",):
            left.spines[side].set_visible(False)
            twin.spines[side].set_visible(False)
    out = {"steps": len(steps), "last_step": steps[-1] if steps else None, "eval": summary}
    if summary:
        bars, labels = [], []
        for name, s in summary["summary"].items():
            bars.append(s["mean_bits"])
            labels.append(f"oracle\n{name.replace('heldout_', 'held-out ')}")
            for b, v in s["baselines"].items():
                bars.append(v)
                labels.append(f"{b}\n{name.replace('heldout_', 'held-out ')}")
        right.bar(range(len(bars)), bars, color=["#1f5fa8" if lab.startswith("oracle") else "#9ca3af" for lab in labels])
        right.set_xticks(range(len(bars)), labels, rotation=0, fontsize=11)
        right.set_ylabel("S (bits), evaluation seed")
        right.spines["top"].set_visible(False)
        right.spines["right"].set_visible(False)
    fig.tight_layout()
    path = Path(a.out) if a.out else run / "report.png"
    fig.savefig(path, dpi=130)
    out["figure"] = str(path)
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
