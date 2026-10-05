"""Plot native discovery measurements; no model fitting or heldout access."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
source = BASE / "DISCOVERY_NATIVE.json"
report = json.loads(source.read_text())
assert len(report["results"]) == 64
panels = []
summary = {"scope": "Native behavioral discovery data only; no recovered internal mechanism or heldout result.",
           "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "panels": []}
for wording in (0, 1):
    pairs = {}
    for row in report["results"]:
        factors = json.loads(row["variant"])
        if factors["update_wording"] != wording:
            continue
        now = factors.pop("query_now")
        key = json.dumps(factors, sort_keys=True)
        assert now not in pairs.setdefault(key, {})
        pairs[key][now] = row
    assert len(pairs) == 16 and all(set(pair) == {0, 1} for pair in pairs.values())
    values = np.array([[pair[t]["metrics"]["target_minus_foil_log_odds"] for t in (0, 1)]
                       for _, pair in sorted(pairs.items())])
    # Establish that every paired prompt differs only by the final added word.
    for pair in pairs.values():
        assert pair[1]["prompt"] == pair[0]["prompt"] + " now"
        assert pair[0]["target_token"] == pair[1]["target_token"]
        assert pair[0]["foil_token"] == pair[1]["foil_token"]
    panels.append(values)
    summary["panels"].append({"update_wording": wording, "pairs": 16,
        "mean_log_odds_without_now": float(values[:, 0].mean()),
        "mean_log_odds_with_now": float(values[:, 1].mean()),
        "mean_paired_change": float(np.diff(values, axis=1).mean()),
        "pairs_increasing_updated_preference": int(np.sum(values[:, 1] > values[:, 0])),
        "updated_beats_old_without_now": int(np.sum(values[:, 0] > 0)),
        "updated_beats_old_with_now": int(np.sum(values[:, 1] > 0))})

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
fig.patch.set_facecolor("#f8fafc")
fig.subplots_adjust(left=.10, right=.96, bottom=.28, top=.72, wspace=.15)
fig.text(.065, .94, "“Now” can favor the old answer", fontsize=25, weight="bold", color="#182638")
fig.text(.065, .885, "Qwen3–0.6B  ·  64 counterbalanced discovery prompts", fontsize=12, color="#526176")
fig.text(.065, .852, "Both prompts state an updated color. Adding “now” to the final question\n"
         "has opposite average effects depending on how the update was phrased.", fontsize=12, color="#26384c", linespacing=1.5, va="top")

low = min(float(x.min()) for x in panels) - .35
high = max(float(x.max()) for x in panels) + .45
titles = ['Update: “…color is now green.”', 'Update: “…color has changed to green.”']
colors = ["#bd542f", "#007f83"]
for i, (ax, values, color) in enumerate(zip(axes, panels, colors)):
    ax.set_facecolor("#ffffff")
    ax.axhspan(0, high, color="#eaf5f1", zorder=0)
    ax.axhline(0, color="#77889a", linewidth=1, linestyle=(0, (4, 3)))
    for pair in values:
        ax.plot([0, 1], pair, color="#a4adb9", alpha=.5, linewidth=1, marker="o", markersize=3, zorder=2)
    means = values.mean(axis=0)
    ax.plot([0, 1], means, color=color, linewidth=3.3, marker="o", markersize=9, zorder=5)
    for x, y in enumerate(means):
        ax.annotate(f"Mean {y:+.2f}", (x, y), xytext=(0, 17 if i else -25),
                    textcoords="offset points", ha="center", color=color, weight="bold", fontsize=11,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 2, "alpha": .88}, zorder=6)
    ax.set_title(titles[i], loc="left", fontsize=12, weight="bold", pad=18)
    ax.set_xlim(-.25, 1.25)
    ax.set_ylim(low, high)
    ax.set_xticks([0, 1], ['“…color is”', '“…color is now”'])
    ax.set_xlabel("Final question ending", labelpad=12, color="#526176")
    ax.tick_params(axis="both", length=0, labelcolor="#344357")
    ax.spines["left"].set_color("#d4dbe3")
    ax.spines["bottom"].set_color("#d4dbe3")
    counts = summary["panels"][i]
    ax.text(.5, -.28, f"Updated color beats old color: {counts['updated_beats_old_without_now']}/16 → {counts['updated_beats_old_with_now']}/16",
            transform=ax.transAxes, ha="center", fontsize=11, weight="bold", color=color)

axes[0].set_ylabel("Preference for updated over old color\nlog probability ratio (nats)", labelpad=12, color="#344357")
axes[1].text(.97, .98, "Above zero: updated color favored", transform=axes[1].transAxes,
             ha="right", va="top", fontsize=9, color="#507568")
fig.text(.065, .13, "Thin lines: 16 matched prompt pairs per panel. Thick line: mean. Each pair differs only by the final “now”.",
         fontsize=10, color="#526176", va="top")
fig.text(.065, .097, "Native behavior only. No internal mechanism or held-out prediction has been established.\n"
         "Counts compare the two color tokens; they are not full-vocabulary accuracy. No statistical error bars are claimed.",
         fontsize=10, color="#526176", linespacing=1.45, va="top")
for suffix in ("png", "pdf"):
    fig.savefig(BASE / f"wording_effect.{suffix}", dpi=180, facecolor=fig.get_facecolor())
(BASE / "WORDING_EFFECT_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print(BASE / "wording_effect.pdf")
