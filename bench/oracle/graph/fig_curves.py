"""Curve areas against a reference (#2951 graph oracle): per question, each method's curve area (score.key: the mean KL
of the model's prediction from the graph over log-uniform description length) against the reference's (by default
VPD's ranked answer); left, every question as a point (below the diagonal: better than the reference); right, a few
questions' curves (the lowest KL so far at each description length).

  fig_curves.py OUT.png --method "LABEL=EVAL_SAMPLES.jsonl:PROGRAM_PREFIX" [--method ...] [--reference vpd] [--examples 3]
The reference program's rows come from the first file (methods', then --baselines) with a valid score for each question.
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

PALETTE = ["#2a78b5", "#d98a1e", "#1d8a6b", "#8a5cc4"]  # methods in order (passes the colour-blind checks)
REFERENCE = "#7f7f7f"


def best(rows, prefix):
    """Per question, the row of `prefix` with the lowest curve area."""
    out = {}
    for r in rows:
        if r["program"].startswith(prefix) and r["score"].get("valid", True) and r["score"].get("curve"):
            if r["behavior"] not in out or score.key(r["score"]) < score.key(out[r["behavior"]]["score"]):
                out[r["behavior"]] = r
    return out


def running(curve, hi):
    pts, low = [], math.inf
    for b, k in sorted(curve):
        low = min(low, k)
        pts.append((max(b, 1.0), low))
    return pts + [(hi, low)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--method", action="append", default=[])
    ap.add_argument("--reference", default="vpd", help="the program on the x axis")
    ap.add_argument("--reference-label", default="VPD's ranked answer")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--baselines", action="append", default=[], help="more eval_samples files to take the reference from")
    a = ap.parse_args()
    methods, ref = [], {}
    for spec in a.method:
        label, rest = spec.split("=", 1)
        path, prefix = rest.rsplit(":", 1)
        rows = [json.loads(x) for x in open(path)]
        methods.append((label, best(rows, prefix)))
        ref = {**best(rows, a.reference), **ref}
    for path in a.baselines:
        ref = {**best([json.loads(x) for x in open(path)], a.reference), **ref}
    area = lambda r: score.key(r["score"])[1]  # noqa: E731

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.25]})
    lo, hi = math.inf, 0.0
    series = [(label, m, PALETTE[j % len(PALETTE)]) for j, (label, m) in enumerate(methods)]
    for label, m, color in series:
        qs = [q for q in m if q in ref]
        if not qs:
            continue
        xs, ys = [area(ref[q]) for q in qs], [area(m[q]) for q in qs]
        lo, hi = min(lo, *xs, *ys), max(hi, *xs, *ys)
        below = sum(y < x for x, y in zip(xs, ys))
        ax.scatter(xs, ys, s=44, color=color, alpha=0.85, edgecolor="white", linewidth=0.6, label=f"{label}: better on {below}/{len(qs)}")
    ax.plot([lo * 0.8, hi * 1.25], [lo * 0.8, hi * 1.25], color=REFERENCE, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{a.reference_label}: curve area (bits)", fontsize=15)
    ax.set_ylabel("curve area (bits)", fontsize=15)
    ax.legend(fontsize=12, frameon=False, loc="upper left")

    common = [q for q in ref if all(q in m for _, m in methods)]
    for q in sorted(common, key=lambda q: area(ref[q]))[::max(1, len(common) // max(a.examples, 1))][:a.examples]:
        for label, m, color in [(a.reference_label, ref, REFERENCE)] + series:
            s = m[q]["score"]
            pts = running(s["curve"], s["hi"])
            bx.step([p[0] for p in pts], [p[1] for p in pts], where="post", color=color, lw=2, alpha=0.85)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlabel("description length (bits)", fontsize=15)
    bx.set_ylabel("KL of the prediction (bits)", fontsize=15)
    for label, color in [(a.reference_label, REFERENCE)] + [(lab, col) for lab, _, col in series]:
        bx.plot([], [], color=color, lw=2, label=label)
    bx.legend(fontsize=12, frameon=False)
    from matplotlib.ticker import FuncFormatter

    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (ax, bx):
        for w in (axis.xaxis, axis.yaxis):
            w.set_major_formatter(plain)
            w.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        axis.tick_params(labelsize=13)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130, facecolor="white")
    for label, m in methods:
        qs = [q for q in m if q in ref]
        print(f"{label}: {len(qs)} questions, median area {sorted(area(m[q]) for q in qs)[len(qs) // 2]:.2f}, "
              f"{a.reference} {sorted(area(ref[q]) for q in qs)[len(qs) // 2]:.2f}, better on {sum(area(m[q]) < area(ref[q]) for q in qs)}")


if __name__ == "__main__":
    main()
