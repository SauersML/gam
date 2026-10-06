"""A shared body and its calls, and the code length F of each fitted stage (mpd_library_bodies_2951)."""
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path, help="the driver's OUT directory (its SUMMARY.json)")
    parser.add_argument("figure", type=Path)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    summary = json.loads((args.run / "SUMMARY.json").read_text())
    stages = [s for s in summary["stages"] if s.get("objective_bits") is not None]
    accepted = {d["stage"] for d in summary["decisions"] if d.get("accepted")}
    calls = summary.get("calls", [])
    readings = {r["call"]: r for r in summary.get("readings", [])}
    evidence = {e["call"]: e for e in summary.get("evidence", [])}
    plt.rcParams.update({"font.size": 18})
    figure, (left, right) = plt.subplots(1, 2, figsize=(18, 7.5), gridspec_kw={"width_ratios": [1, 1.4]})
    figure.patch.set_facecolor("white")

    # F per stage: the base, then each transaction, accepted ones dark.
    names = [s["stage"] for s in stages]
    values = [s["objective_bits"] for s in stages]
    colours = ["#4c4c4c" if n == "base" else ("#1f6fb4" if n in accepted else "#b8c7d6") for n in names]
    left.bar(range(len(values)), values, color=colours)
    for i, v in enumerate(values):
        left.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=14)
    left.set_xticks(range(len(values)), ["M's library" if n == "base" else n for n in names], rotation=30, ha="right")
    left.set_ylabel("code length F (bits)")
    left.spines[["top", "right"]].set_visible(False)

    # The most called body: its calls at their layers, with the heads each reads and the tokens
    # each writes towards.
    bodies = {}
    for call in calls:
        bodies.setdefault(call["body"], []).append(call)
    right.axis("off")
    if bodies:
        body, members = max(bodies.items(), key=lambda kv: len(kv[1]))
        units = max(len(c["replaced"]) for c in members)
        right.add_patch(FancyBboxPatch((0.38, 0.42), 0.24, 0.16, boxstyle="round,pad=0.02", fc="#1f6fb4", ec="none"))
        right.text(0.5, 0.5, f"one body\n{units} units", ha="center", va="center", color="white", fontsize=20)
        for k, call in enumerate(sorted(members, key=lambda c: c["layer"])):
            y = 0.85 - 0.7 * k / max(1, len(members) - 1)
            heads = ", ".join(h for h, _ in evidence.get(call["name"], {}).get("heads", [])[:2])
            writes = readings.get(call["name"], {}).get("writes", [])
            promoted = ", ".join(str(t["token"]) for t in writes[0][0][:3]) if writes else ""
            right.add_patch(FancyBboxPatch((0.0, y - 0.07), 0.24, 0.14, boxstyle="round,pad=0.02", fc="#eef3f8", ec="#1f6fb4"))
            right.text(0.12, y, f"layer {call['layer']} MLP\n{len(call['replaced'])} functions", ha="center", va="center")
            right.add_patch(FancyArrowPatch((0.25, y), (0.37, 0.5), arrowstyle="-|>", mutation_scale=20, color="#4c4c4c"))
            right.text(0.66, y, f"reads {heads}\nwrites towards tokens {promoted}", ha="left", va="center", fontsize=15)
    figure.tight_layout()
    figure.savefig(args.figure, dpi=150, facecolor="white")
    if args.open:
        subprocess.run(["open", "-a", "Preview", str(args.figure)], check=False)


if __name__ == "__main__":
    main()
