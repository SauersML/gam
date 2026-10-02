"""E1: bits per token to say which VPD pieces are on, coded with more and more context (#2951)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

d = json.load(open(Path.home() / "mpd-data/frontier/e12_vpd_only.json"))
E = d["E1_bits_per_token"]
INK, INK2, MUTED, SURF, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE, LIGHT = "#2a78d6", "#9ec5f4"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 15})
SPREAD = 1e6
rows = [("independent", "no context"),
        ("marginal", "knowing how often\neach piece is on"),
        ("previous", "+ the previous\nword's pieces"),
        ("token", "+ which word it is"),
        ("token_and_previous", "both")]

fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
H = 0.38
STORED = {"marginal": f"storing each piece's on-rate ({d['pieces_total']:,} numbers)",
          "previous": "+ each piece's chance of staying on",
          "token": "+ each piece's on-rate for every word in the vocabulary",
          "token_and_previous": "both of the above"}
for i, (k, label) in enumerate(rows):
    y = len(rows) - 1 - i
    data, tables = E[k]["data"], E[k]["model_bits"] / SPREAD
    ax.barh(y, data, height=H, color=BLUE, zorder=3)
    if tables > 0:
        ax.barh(y, tables, left=data + 4, height=H, color=LIGHT, zorder=3)
    total = data + tables
    if k in STORED:
        ax.annotate(STORED[k], (total + 4, y), xytext=(62, 0), textcoords="offset points", va="center",
                    fontsize=12.5, color=INK2, bbox=dict(boxstyle="square,pad=0.15", fc=SURF, ec="none"))
    ax.annotate(f"{total:,.0f}", (total + 4, y), xytext=(8, 0), textcoords="offset points", va="center",
                fontsize=14, color=INK, fontweight="medium" if k in ("independent", "token_and_previous") else "normal")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([l for _, l in rows][::-1], fontsize=14, color=INK)
ax.tick_params(axis="y", length=0, pad=12)
ax.tick_params(axis="x", colors=INK2, labelsize=12.5)
ax.set_xlim(0, 2000)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.set_xlabel("bits needed per word to say which pieces are on", color=INK, labelpad=10)
fig.legend(handles=[Patch(color=BLUE, label="listing which pieces are on, word by word"),
                    Patch(color=LIGHT, label="one-time cost of storing those numbers, spread over a million words")],
           loc="upper left", bbox_to_anchor=(0.2, 0.87), ncol=2, frameon=False, fontsize=13, labelcolor=INK,
           handlelength=1.2, columnspacing=2.2)
fig.suptitle("Which VPD pieces are on for each word is about a third predictable from context", color=INK, fontsize=20,
             fontweight="bold", x=0.025, ha="left", y=0.975)
tok32 = {k: E[k]["total"] for k in ("token", "token_and_previous")}
fig.text(0.025, 0.9, "the stored numbers are paid for once and spread over a million words; "
         f"over the {d['coded_tokens'] // 1000}k words tested, the word-based versions cost more", color=INK2, fontsize=13.5, ha="left")
fig.subplots_adjust(left=0.25, right=0.94, top=0.79, bottom=0.12)
out = Path.home() / "mpd-data/figures/e1_context_coding.png"
fig.savefig(out, facecolor=SURF)
print(out)
