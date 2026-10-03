"""E4 side effects (#2951): what the VPD one-subcomponent emoticon edit and LoRA each disturb, at equal edit success.
Reads e4_side_effects.json written by bench/e4_side_effects_data.py summarize; writes four figures to ~/mpd-data/figures/.

usage: MPD_MEM_GIB=1 e4_side_effects_fig.py [JSON] [HEADLINE_LORA]   (default lora282_lam10: 282 examples, lambda 10)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "mpd-data/frontier/e4_side/e4_side_effects.json"
d = json.load(open(src))
HEAD = sys.argv[2] if len(sys.argv) > 2 else "lora282_lam10"
VHEAD = "vpd_match_" + HEAD
OUT = Path.home() / "mpd-data/figures"
INK, INK2, MUTED, SURF, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#ffffff", "#e8e8e8", "#c3c2b7"
ORANGE, BLUE, BAND = "#eb6834", "#2a78d6", "#f4f3ef"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 17,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
meta = d["meta"]
pf = meta[HEAD]["p_fire"]
lora_name = lambda nm: f"LoRA ({meta[nm]['n_train']} examples, λ = {meta[nm]['lambda']:g})"
HL, HV = lora_name(HEAD), f"VPD edit (strength {meta[VHEAD]['alpha']:.2f})"
MATCH = f"both edits write “o” after an emoticon colon with probability {pf:.1%}"
FAMILY_ORDER = ["Colons and semicolons that are not emoticons", "The letter o elsewhere", "Closing brackets and quotes",
                "Copying from earlier in the text", "Numbers", "Grammar and common words", "Code, web addresses and lists",
                "Everything"]


def style(ax, grid_axis="x"):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(False)
    ax.tick_params(labelsize=15)


def logfmt(ax, axis="x"):
    a = ax.xaxis if axis == "x" else ax.yaxis
    lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
    a.set_major_locator(LogLocator(base=10, subs=(1, 2, 5) if hi / lo < 300 else (1,), numticks=40))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def show_tok(t):
    return "“" + t.replace("\n", "⏎").replace("\t", "⇥") + "”"


def pct(p):
    return f"{p:.0%}" if p >= 0.01 else "<1%"


# ---------------------------------------------------------------- 1. every check, ranked within its family
checks = [c for c in d["checks"] if c["n"] >= 30]
rows = []
for fam in FAMILY_ORDER:
    fc = sorted([c for c in checks if c["family"] == fam], key=lambda c: -c["pairs"][HEAD]["kl_ratio"][0])
    if fc:
        rows.append(("family", fam))
        rows += [("check", c) for c in fc]
emo = next(c for c in d["checks"] if c["key"] == "emoticon") if any(c["key"] == "emoticon" for c in d["checks"]) else None
H = 0.34 * len(rows) + 3.2
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(21, H), dpi=170, gridspec_kw={"width_ratios": [1.15, 1, 1], "wspace": 0.05})
fig.patch.set_facecolor(SURF)
ys = np.arange(len(rows))[::-1]
for ax in (a1, a2, a3):
    style(ax)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks([])
    for y, (kind, _) in zip(ys, rows):
        if kind == "family":
            ax.axhspan(y - 0.5, y + 0.5, color=BAND, zorder=0, lw=0)
ratios = []
for y, (kind, c) in zip(ys, rows):
    if kind == "family":
        a1.text(-0.012, y, c, transform=a1.get_yaxis_transform(), ha="right", va="center", fontsize=14,
                fontweight="bold", color=INK)
        continue
    a1.text(-0.012, y, c["label"], transform=a1.get_yaxis_transform(), ha="right", va="center", fontsize=14, color=INK)
    a3.text(1.01, y, f"{c['n']:,}", transform=a3.get_yaxis_transform(), ha="left", va="center", fontsize=14, color=INK2)
    for ax, key, off in ((a1, "kl", 0.13), (a3, "loss", 0.13)):
        for nm, col, dy in ((VHEAD, ORANGE, off), (HEAD, BLUE, -off)):
            m, lo, hi = c["models"][nm][key]
            if key == "kl":
                lo, m, hi = max(lo, 1e-7), max(m, 1e-7), max(hi, 1e-7)
            ax.plot([lo, hi], [y + dy, y + dy], color=col, lw=1.6, zorder=3, solid_capstyle="round")
            ax.scatter([m], [y + dy], s=34, color=col, edgecolor=SURF, linewidth=1.2, zorder=4)
    r, lo, hi = c["pairs"][HEAD]["kl_ratio"]
    ratios.append((c, r, lo, hi))
    col = BLUE if hi < 1 else ORANGE if lo > 1 else MUTED
    a2.plot([lo, hi], [y, y], color=col, lw=2, zorder=3, solid_capstyle="round")
    a2.scatter([r], [y], s=40, color=col, edgecolor=SURF, linewidth=1.2, zorder=4)
a1.set_xscale("log")
logfmt(a1)
a1.set_xlabel("disturbance: KL from the original model (nats per word, log scale)", fontsize=14)
a1.set_title("How much each edit disturbs the prediction", loc="left", fontsize=14, fontweight="bold", color=INK, pad=10)
a2.set_xscale("log")
logfmt(a2)
a2.axvline(1, color=INK2, lw=1, zorder=2)
a2.set_xlabel("LoRA's disturbance ÷ VPD's (log scale)", fontsize=14)
a2.set_title("Ratio: left of 1, LoRA disturbs less", loc="left", fontsize=14, fontweight="bold", color=INK, pad=10)
a3.axvline(0, color=INK2, lw=1, zorder=2)
a3.set_xscale("symlog", linthresh=1e-3, linscale=0.6)
a3.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
a3.set_xlabel("change in loss on the correct next word (nats; right = worse)", fontsize=14)
a3.set_title("Effect on getting the right answer", loc="left", fontsize=14, fontweight="bold", color=INK, pad=10)
a3.text(1.01, len(rows) - 0.3, "positions", transform=a3.get_yaxis_transform(), ha="left", va="bottom", fontsize=14, color=INK2)
n_l = sum(hi < 1 for _, _, _, hi in ratios)
n_v = sum(lo > 1 for _, _, lo, _ in ratios)
other = f"VPD less on {n_v}" if n_v else "none the other way"
fig.suptitle(f"At equal edit success, LoRA disturbs {n_l} of {len(ratios)} abilities less than the VPD edit; {other}",
             x=0.01, ha="left", y=1 - 0.35 / H, fontsize=26, fontweight="bold", color=INK)
fig.legend(handles=[Line2D([], [], color=ORANGE, marker="o", lw=1.6, ms=6, mec=SURF, label=HV),
                    Line2D([], [], color=BLUE, marker="o", lw=1.6, ms=6, mec=SURF, label=HL),
                    Line2D([], [], color=MUTED, marker="o", lw=2, ms=6, mec=SURF, label="ratio interval includes 1")],
           loc="upper left", bbox_to_anchor=(0.3, 1 - 0.95 / H), ncol=3, frameon=False, fontsize=16)
fig.subplots_adjust(left=0.3, right=0.955, top=1 - 2.2 / H, bottom=0.7 / H)
fig.savefig(OUT / "e4_side_effects_checks.png", facecolor=SURF)
plt.close(fig)

# ---------------------------------------------------------------- 2. where the disturbance sits
bd = d["breakdowns"]
fig, axes = plt.subplots(2, 3, figsize=(21, 12.5), dpi=170)
fig.patch.set_facecolor(SURF)
panels = [("distance", "Distance from the nearest earlier ‘:’ or ‘;’"), ("input_token", "What the current token is"),
          ("emoticon_context", "Whether the document contains an emoticon"),
          ("confidence", "How confident the original model was"), ("frequency", "How common the correct next word is")]
for ax, (axis, title) in zip(axes.ravel(), panels):
    style(ax, "y")
    bins = [b for b in bd[axis] if b["n"] >= 30]
    x = np.arange(len(bins))
    for nm, col, dx in ((VHEAD, ORANGE, -0.17), (HEAD, BLUE, 0.17)):
        m = np.array([max(b["models"][nm]["kl"][0], 1e-8) for b in bins])
        lo = np.array([max(b["models"][nm]["kl"][1], 1e-8) for b in bins])
        hi = np.array([max(b["models"][nm]["kl"][2], 1e-8) for b in bins])
        ax.vlines(x + dx, lo, hi, color=col, lw=2, zorder=3)
        ax.scatter(x + dx, m, s=46, color=col, edgecolor=SURF, linewidth=1.4, zorder=4)
    ax.set_yscale("log")
    logfmt(ax, "y")
    ax.set_xticks(x)
    short = {"no emoticon earlier in the document": "no emoticon earlier", "an emoticon earlier in the document": "emoticon earlier"}
    labs = [short.get(b["label"], b["label"]) for b in bins]
    rot = 30 if sum(len(s) for s in labs) > 60 else 0
    ax.set_xticklabels(labs, rotation=rot, ha="right" if rot else "center", fontsize=14)
    for xi, b in zip(x, bins):
        ax.text(xi, 1.0, f"{b['n']:,}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=11, color=MUTED)
    ax.set_ylabel("disturbance (KL, nats per word)", fontsize=14)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=INK, pad=20)
ax = axes[1, 2]
style(ax, "both")
for nm, col in ((VHEAD, ORANGE), (HEAD, BLUE)):
    lz = d["lorenz"][nm]
    ax.plot(lz["frac"], lz["share"], color=col, lw=2.4)
ax.set_xscale("log")
logfmt(ax)
ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1,), numticks=20))
ax.set_xlim(1e-6, 1)
ax.set_ylim(0, 1.02)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4%}".rstrip("0").rstrip(".") if v < 0.01 else f"{v:.0%}"))
ax.set_xlabel("most-disturbed fraction of all positions", fontsize=14)
ax.set_ylabel("share of all disturbance", fontsize=14)
ax.set_title("How concentrated the disturbance is", loc="left", fontsize=14, fontweight="bold", color=INK, pad=20)
fig.legend(handles=[Line2D([], [], color=ORANGE, marker="o", lw=2, ms=7, mec=SURF, label=HV),
                    Line2D([], [], color=BLUE, marker="o", lw=2, ms=7, mec=SURF, label=HL)],
           loc="upper left", bbox_to_anchor=(0.01, 0.945), ncol=2, frameon=False, fontsize=17)
near = [b for b in bd["distance"][:4]]
share = lambda nm, bs: sum(b["models"][nm]["kl_sum"] for b in bs) / sum(b["models"][nm]["kl_sum"] for b in bd["distance"])
fig.suptitle(f"{share(VHEAD, near):.0%} of the VPD edit's disturbance is on or within 3 tokens after a ‘:’ or ‘;’; "
             f"{share(HEAD, near):.0%} of LoRA's", x=0.01, ha="left", fontsize=26, fontweight="bold", color=INK, y=0.985)
fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.1, hspace=0.62, wspace=0.22)
fig.savefig(OUT / "e4_side_effects_breakdown.png", facecolor=SURF)
plt.close(fig)

# ---------------------------------------------------------------- 3. the most-disturbed non-emoticon contexts
def clip_ctx(s, n=58):
    s = s.replace("\n", "⏎").replace("\t", "⇥")
    return ("…" + s[-n:]) if len(s) > n else s


fig = plt.figure(figsize=(21, 24), dpi=170)
fig.patch.set_facecolor(SURF)
fig.suptitle("The 20 most-disturbed positions in ordinary text, one per document, emoticons excluded",
             x=0.01, ha="left", y=0.995, fontsize=24, fontweight="bold", color=INK)
cols = [0.01, 0.47, 0.585, 0.745, 0.905]
for k, (nm, title, col) in enumerate(((VHEAD, HV, ORANGE), (HEAD, HL, BLUE))):
    top = 0.95 - k * 0.475
    fig.text(0.01, top, title, fontsize=16, fontweight="bold", color=col, ha="left", va="top")
    hy = top - 0.022
    for x, h in zip(cols, ("context", "true next word", "original model's guess", "edited model's guess", "disturbance (KL)")):
        fig.text(x, hy, h, fontsize=14, color=INK2, ha="left", va="top", fontweight="bold")
    for i, e in enumerate(d["examples"][nm]):
        y = hy - 0.021 * (i + 1)
        if i % 2 == 0:
            fig.patches.append(plt.Rectangle((0.005, y - 0.0145), 0.99, 0.0195, transform=fig.transFigure,
                                             color=BAND, zorder=0, lw=0))
        ctx = clip_ctx(e["context"])
        t1 = fig.text(cols[0], y, ctx, fontsize=14, color=INK2, ha="left", va="top", family=["Menlo", "Arial Unicode MS"])
        fig.canvas.draw() if i == 0 else None
        bb = t1.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        fig.text(bb.x1 + 0.001, y, e["token"].replace("\n", "⏎").replace("\t", "⇥"), fontsize=14, color=INK,
                 fontweight="bold", ha="left", va="top", family=["Menlo", "Arial Unicode MS"])
        fig.text(cols[1], y, show_tok(e["true_next"]), fontsize=14, color=INK, ha="left", va="top")
        fig.text(cols[2], y, f"{show_tok(e['orig_top'])} {pct(e['orig_p'])}", fontsize=14, color=INK, ha="left", va="top")
        fig.text(cols[3], y, f"{show_tok(e['edit_top'])} {pct(e['edit_p'])}", fontsize=14,
                 color=col if e["edit_top"] != e["orig_top"] else INK, ha="left", va="top",
                 fontweight="bold" if e["edit_top"] != e["orig_top"] else "normal")
        fig.text(cols[4], y, f"{e['kl']:.3g}", fontsize=14, color=INK, ha="left", va="top")
fig.savefig(OUT / "e4_side_effects_examples.png", facecolor=SURF)
plt.close(fig)

# ---------------------------------------------------------------- 4. the ratio for every matched pair
pairs = [p[0] for p in d["pairs"]]
pairs.sort(key=lambda nm: (meta[nm]["n_train"] != 282, meta[nm]["lambda"]))
cr = [c for _, c in rows if not isinstance(c, str)]
Z = np.array([[np.log2(c["pairs"][nm]["kl_ratio"][0]) for nm in pairs] for c in cr])
sig = np.array([[(c["pairs"][nm]["kl_ratio"][2] < 1) or (c["pairs"][nm]["kl_ratio"][1] > 1) for nm in pairs] for c in cr])
H4 = 0.3 * len(cr) + 3.6
fig, ax = plt.subplots(figsize=(15, H4), dpi=170)
fig.patch.set_facecolor(SURF)
cmap = LinearSegmentedColormap.from_list("bo", [BLUE, "#f2f1ec", ORANGE])
lim = np.nanmax(np.abs(Z[np.isfinite(Z)]))
im = ax.imshow(np.clip(Z, -lim, lim), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        r = 2 ** Z[i, j]
        ax.text(j, i, (f"{r:.2g}×" if r < 10 else f"{r:.0f}×"), ha="center", va="center", fontsize=12,
                color=INK if sig[i, j] else MUTED, fontweight="bold" if sig[i, j] else "normal")
ax.set_yticks(range(len(cr)))
ax.set_yticklabels([c["label"] for c in cr], fontsize=14)
ax.set_xticks(range(len(pairs)))
ax.set_xticklabels([f"{meta[nm]['n_train']} examples\nλ = {meta[nm]['lambda']:g}\nsuccess {meta[nm]['p_fire']:.1%}" for nm in pairs], fontsize=14)
ax.xaxis.tick_top()
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_ticks([-lim, 0, lim])
cb.set_ticklabels([f"LoRA {2 ** lim:.0f}× less", "equal", f"LoRA {2 ** lim:.0f}× more"])
cb.outline.set_visible(False)
fig.suptitle("LoRA's disturbance ÷ VPD's, for every LoRA setting at equal edit success",
             x=0.01, ha="left", fontsize=22, fontweight="bold", color=INK, y=1 - 0.3 / H4)
fig.subplots_adjust(left=0.37, right=0.86, top=1 - 2.4 / H4, bottom=0.2 / H4)
fig.savefig(OUT / "e4_side_effects_all_settings.png", facecolor=SURF)
plt.close(fig)
# ---------------------------------------------------------------- 5. each Pile subset
doms = sorted(d["breakdowns"]["domain"], key=lambda b: -b["models"][VHEAD]["loss"][0])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(21, 0.55 * len(doms) + 2.6), dpi=170, gridspec_kw={"wspace": 0.06})
fig.patch.set_facecolor(SURF)
ys = np.arange(len(doms))[::-1]
for ax, key in ((a1, "loss"), (a2, "kl")):
    style(ax)
    for y, b in zip(ys, doms):
        for nm, col, dy in ((VHEAD, ORANGE, 0.15), (HEAD, BLUE, -0.15)):
            m, lo, hi = b["models"][nm][key]
            ax.plot([lo, hi], [y + dy, y + dy], color=col, lw=2.6, solid_capstyle="round", zorder=3)
            ax.scatter([m], [y + dy], s=80, color=col, edgecolor=SURF, linewidth=1.5, zorder=4)
    ax.set_yticks(ys)
    ax.set_ylim(-0.6, len(doms) - 0.4)
a1.set_yticklabels([f"{b['label']} ({d['domain_docs'][b['label']]} docs)" for b in doms], color=INK)
a2.set_yticklabels([])
a1.axvline(0, color=INK2, lw=1.2)
a1.set_xlabel("change in loss per word (nats; right = worse)")
a2.set_xscale("log")
logfmt(a2)
a2.set_xlabel("disturbance: KL from the original model (nats per word)")
a1.set_title("Loss", loc="left", fontweight="bold", color=INK, pad=10)
a2.set_title("Disturbance", loc="left", fontweight="bold", color=INK, pad=10)
H5 = 0.55 * len(doms) + 2.6
fig.legend(handles=[Line2D([], [], color=ORANGE, marker="o", lw=2.6, ms=9, mec=SURF, label=HV),
                    Line2D([], [], color=BLUE, marker="o", lw=2.6, ms=9, mec=SURF, label=HL)],
           loc="upper right", bbox_to_anchor=(0.99, 1 - 0.1 / H5), ncol=2, frameon=False, fontsize=17)
fig.suptitle("Each edit's effect on every Pile subset, at equal edit success", x=0.01, ha="left", y=1 - 0.1 / H5,
             fontsize=26, fontweight="bold", color=INK)
fig.subplots_adjust(left=0.22, right=0.98, top=1 - 1.5 / H5, bottom=1.0 / H5)
fig.savefig(OUT / "e4_side_effects_domains.png", facecolor=SURF)
plt.close(fig)

print("\n".join(str(OUT / f) for f in ("e4_side_effects_domains.png", "e4_side_effects_checks.png", "e4_side_effects_breakdown.png",
                                       "e4_side_effects_examples.png", "e4_side_effects_all_settings.png")))
