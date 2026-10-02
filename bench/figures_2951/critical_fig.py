"""VPD components switched on per word against what the model predicts at that word and how far VPD's
prediction is from the model's (#2951). Data from critical_data.py."""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

d = np.load(Path.home() / "mpd-data/figures/data/critical.npz")
tok = Tokenizer.from_file(str(Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"))
INK, INK2, MUTED, SURF, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 15})
keep = np.ones(d["pieces"].shape, bool)
keep[:, 0] = False  # the first word of a chunk is an outlier of its own (535 components)
pieces = d["pieces"][keep].astype(float)
ids = d["ids"][keep]
mean = pieces.mean()
pct = lambda e: f"{100 * e:g}%"
# (title, values, bin edges left to right, tick label per bin, gloss)
panels = [("probability the model puts on its\ntop guess for the next word", d["top"][keep],
           [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0], None, None),
          ("probability the model gave the word\nthat actually came next", np.exp(-d["loss"][keep]),
           [1.0, 0.9, 0.5, 0.2, 0.05, 0.01, 0.0], None, None),
          ("KL divergence between the model's and\nVPD's next-word probabilities (nats)", d["kl"][keep],
           [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, math.inf], None, "0 = identical predictions")]


def bin_labels(edges):
    falling = edges[0] > edges[-1]
    out = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        if falling:
            out.append(f"≥ {pct(b)}" if i == 0 else (f"< {pct(a)}" if i == len(edges) - 2 else f"{pct(b)}–{pct(a)}"))
        else:
            out.append(f"≥ {a:g}" if math.isinf(b) else f"{a:g}–{b:g}")
    return out


def word(t):
    s = tok.decode([int(t)]).replace("\n", "↵").replace("\t", "⇥").strip()
    return f"“{s}”" if s else "“ ”"


def examples(mask, n=3):
    """Words over-represented in this bin, weighted by how often they occur there."""
    all_c = np.bincount(ids, minlength=ids.max() + 1)
    in_c = np.bincount(ids[mask], minlength=ids.max() + 1)
    share = mask.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(in_c >= 15, in_c * np.log(in_c / (all_c * share)), -np.inf)
    score[SPECIAL] = -np.inf
    return [word(t) for t in np.argsort(-score)[:n] if np.isfinite(score[t])]


SPECIAL = [i for i in range(tok.get_vocab_size()) if (tok.id_to_token(i) or "").startswith("<|")]
fig, axes = plt.subplots(1, 3, figsize=(20, 8.4), dpi=200, sharey=True)
fig.patch.set_facecolor(SURF)
for ax, (title, x, edges, _, gloss) in zip(axes, panels):
    lo, hi = np.minimum(edges[:-1], edges[1:]), np.maximum(edges[:-1], edges[1:])
    nb = len(edges) - 1
    masks = [(x >= lo[i]) & ((x < hi[i]) | (i == 0 and edges[0] > edges[-1])) for i in range(nb)]
    h = [pieces[m].mean() for m in masks]
    ax.set_facecolor(SURF)
    ax.bar(range(nb), h, width=0.62, color=BLUE, zorder=3)
    for i, v in enumerate(h):
        ax.annotate(f"{v:.0f}", (i, v), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                    fontsize=12.5, color=INK, zorder=5)
    ax.axhline(mean, color=INK2, lw=1.1, zorder=4)
    labels = bin_labels(edges)
    ax.set_xticks(range(nb))
    ax.set_xticklabels(labels, fontsize=12, color=INK)
    for i, m in enumerate(masks):
        ax.annotate(f"{int(m.sum()):,} words", (i, 0), xycoords=("data", "axes fraction"), xytext=(0, -30),
                    textcoords="offset points", ha="center", va="top", fontsize=10.5, color=MUTED)
        ax.annotate("\n".join(examples(m)), (i, 0), xycoords=("data", "axes fraction"), xytext=(0, -52),
                    textcoords="offset points", ha="center", va="top", fontsize=11, color=INK2, linespacing=1.35)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", colors=INK2, labelsize=12.5, length=0)
    ax.set_title(title, color=INK, fontsize=14.5, loc="left", pad=14 if gloss is None else 30)
    if gloss:
        ax.text(0, 1.03, gloss, transform=ax.transAxes, fontsize=12, color=INK2, ha="left", va="bottom")
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
axes[0].set_ylabel("VPD components switched on for this word", color=INK, labelpad=10)
axes[0].set_ylim(0, 390)
fig.suptitle("Only the most predictable words get fewer VPD components; the words VPD gets most wrong get the most",
             color=INK, fontsize=19, fontweight="bold", x=0.02, ha="left", y=0.975)
fig.text(0.02, 0.9, f"{len(pieces):,} words of web text through VPD's 4-layer model; the line is the average word "
         f"({mean:.0f} components); under each bar, words that turn up most in it", color=INK2, fontsize=14, ha="left")
fig.subplots_adjust(left=0.055, right=0.99, top=0.74, bottom=0.24, wspace=0.1)
out = Path.home() / "mpd-data/figures/critical_words.png"
fig.savefig(out, facecolor=SURF)
print(out)
