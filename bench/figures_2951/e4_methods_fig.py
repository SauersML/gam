"""E4 edit methods (#2951): off-target disturbance against edit success for every way of writing "o" after an
emoticon colon, and the most-disturbed ordinary positions of the best new method. Reads
bench/e4_edit_methods.py's screen.json and, when present, the full-harness summary of the final edit set.

usage: MPD_MEM_GIB=1 e4_methods_fig.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

M = Path.home() / "mpd-data/frontier/e4_side/methods"
OUT = Path.home() / "mpd-data/figures"
scr = json.load(open(M / "screen.json"))
INK, INK2, MUTED, SURF, AXIS, BAND = "#0b0b0b", "#52514e", "#898781", "#ffffff", "#c3c2b7", "#f4f3ef"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 24, "axes.edgecolor": AXIS,
                     "xtick.color": INK2, "ytick.color": INK2})
best = lambda prefix: min((k for k in scr if k.startswith(prefix) and isinstance(scr[k], list)),
                          key=lambda k: next((q["kl"] for q in scr[k] if q.get("target") == 0.985 and q.get("kl")), np.inf))
SERIES = [  # (family key, label, colour, dashed)
    ("vpd", "VPD subcomponent edit", "#eb6834", False),
    (f"{best('spec')}", "most emoticon-specific VPD subcomponent", "#eda100", False),
    (best("contrast"), "ROME, also avoiding other colons", "#e87ba4", False),
    (best("null"), "AlphaEdit (avoids directions common in text)", "#4a3aa7", False),
    (best("memit"), "MEMIT", "#008300", False),
    ("rome+fisher", "ROME", "#1baf7a", False),
    ("compiled_span8", "least change to ordinary text, exact on emoticons", "#e34948", False),
]
fig, ax = plt.subplots(figsize=(22, 11.5), dpi=150)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
ends = []
for key, label, col, dashed in SERIES:
    pts = [q for q in scr[key] if q.get("kl")]
    x, y = [q["p_fire"] for q in pts], [q["kl"] for q in pts]
    ax.plot(x, y, color=col, lw=3, zorder=3, solid_capstyle="round", ls=(0, (4, 2)) if dashed else "-")
    ax.scatter(x, y, s=70, color=col, edgecolor=SURF, linewidth=1.5, zorder=4)
    ends.append([np.log10(y[-1]), label, col])
ends.sort()
for i in range(1, len(ends)):  # labels in the right margin, at least 0.13 decades apart
    ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.13)
for ly, label, col in ends:
    ax.text(1.01, 10 ** ly, label, color=col, fontsize=19, va="center", fontweight="bold",
            transform=ax.get_yaxis_transform(), clip_on=False)
loras = {k: v[0] for k, v in scr.items() if k.startswith("lora") and not k.startswith("loraneg") and isinstance(v, list)}
ax.scatter([v["p_fire"] for v in loras.values()], [v["kl"] for v in loras.values()], s=170, color="#2a78d6",
           edgecolor=SURF, linewidth=2, zorder=5)
lb = max(loras.values(), key=lambda v: v["p_fire"])
ax.text(lb["p_fire"] - 0.002, lb["kl"] * 1.25, "LoRA (8 settings)", color="#2a78d6", fontsize=19, fontweight="bold", ha="right")
negs = {k: v[0] for k, v in scr.items() if k.startswith("loraneg")}
if negs:
    ax.scatter([v["p_fire"] for v in negs.values()], [v["kl"] for v in negs.values()], s=190, marker="D",
               facecolor=SURF, edgecolor="#2a78d6", linewidth=3, zorder=5)
    nb = min(negs.values(), key=lambda v: v["kl"])
    ax.text(nb["p_fire"] - 0.002, nb["kl"] * 0.62, "LoRA, trained to spare other colons", color="#2a78d6", fontsize=19,
            fontweight="bold", ha="right")
ax.set_yscale("log")
ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1,), numticks=20))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlim(0.78, 1.0)
ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("edit success: probability of “o” after an emoticon colon")
ax.set_ylabel("disturbance of ordinary text (KL, nats per word)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.suptitle("Off-target disturbance against edit success, every edit method", x=0.01, ha="left", y=0.985,
             fontsize=32, fontweight="bold", color=INK)
fig.subplots_adjust(left=0.09, right=0.66, top=0.9, bottom=0.1)
fig.savefig(OUT / "e4_methods_frontier.png", facecolor=SURF)
plt.close(fig)
print(OUT / "e4_methods_frontier.png")

# ---------------------------------------------------------------- the full harness at equal edit success
if (M / "table.json").exists():
    T = json.load(open(M / "table.json"))
    keys = [k for k in sorted(T, key=lambda k: -T[k]["kl_all"][0])
            if k != "vpd_at_hardneg" and (k == "compiled_span8" or not k.startswith("compiled")) and not k.startswith("decomp_span")]
    FAMILY = [("#eb6834", "parameter decomposition (VPD subcomponents)"),
              ("#a3360f", "parameter decomposition (program-size)"), ("#2a78d6", "fine-tuning (LoRA)"),
              ("#898781", "direct weight edit"), ("#1baf7a", "direct weight edit, least change to ordinary text")]
    COL = {"vpd": "#eb6834", "specific_subcomponent": "#eb6834", "decomp_own": "#a3360f", "lora": "#2a78d6",
           "lora_hardneg": "#2a78d6", "compiled_span8": "#1baf7a"}
    panels = [("kl_all", "Disturbance of all held-out text", "KL from the original model (nats per word)", True),
              ("kl_spaced_colon", "After a colon that is not an emoticon", "KL from the original model (nats per word)", True),
              ("hellaswag", "HellaSwag", "change in the right answer's log-probability share (nats)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(30, 0.62 * len(keys) + 3.4), dpi=130, sharey=True, gridspec_kw={"wspace": 0.08})
    fig.patch.set_facecolor(SURF)
    ys = np.arange(len(keys))
    for ax, (key, title, xl, logx) in zip(axes, panels):
        for y, k in zip(ys, keys):
            v = T[k]["bench_d_margin"][key] if key == "hellaswag" else T[k][key]
            if v[0] is None or not np.isfinite(v[0]):
                continue
            col = COL.get(k, "#898781")
            ax.plot([v[1], v[2]], [y, y], color=col, lw=3, solid_capstyle="round", zorder=3)
            ax.scatter([v[0]], [y], s=120, color=col, edgecolor=SURF, linewidth=1.8, zorder=4,
                       marker="D" if k == "lora_hardneg" else "o")
        if logx:
            ax.set_xscale("log")
            lo, hi = ax.get_xlim()
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1,) if hi / lo > 100 else (1, 2, 5), numticks=20))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.axvline(0, color=INK2, lw=1.2, zorder=1)
        ax.set_xlabel(xl, fontsize=20)
        ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=12)
        for sd in ("top", "right"):
            ax.spines[sd].set_visible(False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([T[k]["label"] + (f" ({T[k]['p_fire']:.1%} success)" if abs(T[k]["p_fire"] - 0.9848) > 1e-3 else "")
                             for k in keys], color=INK, fontsize=20)
    H = 0.62 * len(keys) + 3.4
    fig.suptitle("Side effects of each edit, all at 98.5% edit success", x=0.01, ha="left", y=1 - 0.15 / H,
                 fontsize=32, fontweight="bold", color=INK)
    fig.legend(handles=[Line2D([], [], color=c, marker="o", lw=3, ms=11, mec=SURF, label=l) for c, l in FAMILY],
               loc="upper left", ncol=5, frameon=False, fontsize=20, bbox_to_anchor=(0.005, 1 - 0.75 / H))
    fig.subplots_adjust(left=0.27, right=0.99, top=1 - 2.2 / H, bottom=1.2 / H)
    fig.savefig(OUT / "e4_methods_matched.png", facecolor=SURF)
    plt.close(fig)
    print(OUT / "e4_methods_matched.png")

# ---------------------------------------------------------------- worst positions of the best new method
full = M / "compiled/e4_side_effects.json"
if full.exists():
    d = json.load(open(full))
    BEST = "compiled_span8"
    ex = d["examples"].get(BEST, [])
    show = lambda t: "“" + t.replace("\n", "⏎").replace("\t", "⇥") + "”"
    pct = lambda p: f"{p:.0%}" if p >= 0.01 else "<1%"
    fig = plt.figure(figsize=(26, 15), dpi=140)
    fig.patch.set_facecolor(SURF)
    fig.suptitle("The least-change edit: its 20 most-changed positions in ordinary text", x=0.01, ha="left", y=0.985,
                 fontsize=30, fontweight="bold", color=INK)
    cols = [0.01, 0.5, 0.63, 0.79, 0.94]
    hy = 0.9
    for x, h in zip(cols, ("context", "true next word", "original guess", "edited guess", "KL")):
        fig.text(x, hy, h, fontsize=18, color=INK2, ha="left", va="top", fontweight="bold")
    for i, e in enumerate(ex[:20]):
        y = hy - 0.042 * (i + 1)
        if i % 2 == 0:
            fig.patches.append(plt.Rectangle((0.005, y - 0.031), 0.99, 0.04, transform=fig.transFigure, color=BAND, lw=0, zorder=0))
        ctx = e["context"].replace("\n", "⏎").replace("\t", "⇥")
        ctx = ("…" + ctx[-46:]) if len(ctx) > 46 else ctx
        fig.text(cols[0], y, ctx + " ", fontsize=17, color=INK2, ha="left", va="top", family=["Menlo", "Arial Unicode MS"])
        fig.text(cols[1] - 0.012, y, e["token"].replace("\n", "⏎"), fontsize=17, color=INK, ha="right", va="top",
                 fontweight="bold", family=["Menlo", "Arial Unicode MS"])
        fig.text(cols[1], y, show(e["true_next"]), fontsize=17, color=INK, ha="left", va="top")
        fig.text(cols[2], y, f"{show(e['orig_top'])} {pct(e['orig_p'])}", fontsize=17, color=INK, ha="left", va="top")
        changed = e["edit_top"] != e["orig_top"]
        fig.text(cols[3], y, f"{show(e['edit_top'])} {pct(e['edit_p'])}", fontsize=17, color="#1baf7a" if changed else INK,
                 ha="left", va="top", fontweight="bold" if changed else "normal")
        fig.text(cols[4], y, f"{e['kl']:.2g}", fontsize=17, color=INK, ha="left", va="top")
    fig.savefig(OUT / "e4_methods_examples.png", facecolor=SURF)
    plt.close(fig)
    print(OUT / "e4_methods_examples.png")
