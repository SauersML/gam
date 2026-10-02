"""E3: how many VPD subcomponents each kind of word switches on (#2951)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

d = json.load(open(Path.home() / "mpd-data/frontier/e123_vpd4l.json"))
E = d["E3"]
INK, INK2, MUTED, SURF, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 15})
NAMES = {"first position": "first word of a text chunk",
         "other": "mixed letters, digits and symbols",
         "content word": "content words (nouns, verbs, …)",
         "word continuation": "the rest of a split-up word",
         "punctuation": "punctuation",
         "function word": "small words (the, of, and, …)",
         "number": "numbers",
         "whitespace/newline": "spaces and line breaks"}
rows = sorted(E.items(), key=lambda kv: kv[1]["vpd"]["pieces_per_token"])
n_all = sum(v["vpd_tokens"] for _, v in rows)
mean = sum(v["vpd"]["pieces_per_token"] * v["vpd_tokens"] for _, v in rows) / n_all

fig, ax = plt.subplots(figsize=(13, 6.6), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
for y, (k, v) in enumerate(rows):
    x = v["vpd"]["pieces_per_token"]
    ax.barh(y, x, height=0.42, color=BLUE, zorder=3)
    t = ax.annotate(f"{x:.0f}", (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=14, color=INK,
                    zorder=5, bbox=dict(boxstyle="square,pad=0.1", fc=SURF, ec="none"))
    ax.annotate(f"{v['vpd_tokens']:,} words", (1, 0.5), xycoords=t, xytext=(10, 0), textcoords="offset points",
                va="center", fontsize=12, color=MUTED, zorder=5, bbox=dict(boxstyle="square,pad=0.1", fc=SURF, ec="none"))
ax.axvline(mean, color=INK2, lw=1.1, zorder=2)
ax.annotate(f"average word: {mean:.0f}", (mean, len(rows) - 0.45), xytext=(6, 0), textcoords="offset points",
            ha="left", va="bottom", fontsize=13, color=INK2)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([NAMES.get(k, k) for k, _ in rows], fontsize=14, color=INK)
ax.tick_params(axis="y", length=0, pad=12)
ax.tick_params(axis="x", colors=INK2, labelsize=12.5)
ax.set_xlim(0, 600)
ax.set_ylim(-0.6, len(rows) - 0.1)
ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.set_xlabel("VPD subcomponents switched on, per word", color=INK, labelpad=10)
fig.suptitle("Which words need the most VPD subcomponents", color=INK, fontsize=21, fontweight="bold",
             x=0.025, ha="left", y=0.975)
fig.text(0.025, 0.895, f"VPD's 4-layer model reading {n_all:,} words of web text, grouped by kind of word",
         color=INK2, fontsize=14, ha="left")
fig.subplots_adjust(left=0.29, right=0.96, top=0.83, bottom=0.11)
out = Path.home() / "mpd-data/figures/e3_token_types.png"
fig.savefig(out, facecolor=SURF)
print(out)
