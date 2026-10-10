"""Held-out curve areas (#2951 graph oracle): per question, each method's curve area (score.key: the mean KL of the
model's prediction from the graph over log-uniform description length) against the search's whole answer; left,
every question as a point (below the diagonal: better than the search); right, a few questions' curves (the lowest
KL so far at each description length).

  fig_curves.py OUT.png --method "LABEL=EVAL_SAMPLES.jsonl:PROGRAM_PREFIX" [--method ...] [--examples 3]
Baselines (the search's whole answer, VPD's answer) come from the first file that has them.
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

PALETTE = ["#2a78b5", "#d98a1e", "#1d8a6b", "#8a5cc4"]  # methods in order (with VPD's #b03a2e: passes the colour-blind checks)
SEARCH, VPD = "#7f7f7f", "#b03a2e"


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
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args()
    methods, base = [], {}
    for spec in a.method:
        label, rest = spec.split("=", 1)
        path, prefix = rest.rsplit(":", 1)
        rows = [json.loads(x) for x in open(path)]
        methods.append((label, best(rows, prefix)))
        for name in ("search_full", "vpd"):
            if name not in base and any(r["program"] == name for r in rows):
                base[name] = best(rows, name)
    search = base["search_full"]
    area = lambda r: score.key(r["score"])[1]  # noqa: E731

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.25]})
    lo, hi = math.inf, 0.0
    series = [("VPD's answer", base.get("vpd", {}), VPD)] + [(label, m, PALETTE[j % len(PALETTE)]) for j, (label, m) in enumerate(methods)]
    for label, m, color in series:
        qs = [q for q in m if q in search]
        if not qs:
            continue
        xs, ys = [area(search[q]) for q in qs], [area(m[q]) for q in qs]
        lo, hi = min(lo, *xs, *ys), max(hi, *xs, *ys)
        below = sum(y < x for x, y in zip(xs, ys))
        ax.scatter(xs, ys, s=44, color=color, alpha=0.85, edgecolor="white", linewidth=0.6, label=f"{label}: below on {below}/{len(qs)}")
    ax.plot([lo * 0.8, hi * 1.25], [lo * 0.8, hi * 1.25], color=SEARCH, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("search's curve area (bits)", fontsize=15)
    ax.set_ylabel("curve area (bits)", fontsize=15)
    ax.legend(fontsize=12, frameon=False, loc="upper left")

    common = [q for q in search if all(q in m for _, m in methods)]
    for q in sorted(common, key=lambda q: area(search[q]))[::max(1, len(common) // max(a.examples, 1))][:a.examples]:
        for label, m, color in [("search", search, SEARCH)] + series:
            if q in m:
                s = m[q]["score"]
                pts = running(s["curve"], s["hi"])
                bx.step([p[0] for p in pts], [p[1] for p in pts], where="post", color=color, lw=2, alpha=0.85)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlabel("description length (bits)", fontsize=15)
    bx.set_ylabel("KL of the prediction (bits)", fontsize=15)
    for label, color in [("search", SEARCH)] + [(lab, col) for lab, _, col in series]:
        bx.plot([], [], color=color, lw=2, label=label)
    bx.legend(fontsize=12, frameon=False)
    for axis in (ax, bx):
        axis.tick_params(labelsize=13)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130, facecolor="white")
    for label, m in methods:
        qs = [q for q in m if q in search]
        print(f"{label}: {len(qs)} questions, median area {sorted(area(m[q]) for q in qs)[len(qs) // 2]:.2f}, "
              f"search {sorted(area(search[q]) for q in qs)[len(qs) // 2]:.2f}, below search on {sum(area(m[q]) < area(search[q]) for q in qs)}")


if __name__ == "__main__":
    main()
