"""Figures of a printed decomposition (#2951), one idea per PNG.

Reads the ``printouts.json`` that ``mpd_printer_2951`` writes for a modular-addition model and renders:

* ``<prefix>_rules.png``: the decomposed program as its rules; each instance a bar of its bits, rules in order.
* ``<prefix>_bits.png``: program bits (algorithm, constants) against the bits of behaviour still unexplained,
  over the amount of behaviour explained.
* ``<prefix>_unresolved.png``: where the last program is not yet the model, input by input (a, b).

    python mpd_printer_figures_2951.py PRINTOUTS_JSON OUT_DIR PREFIX
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "DejaVu Sans"], "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
})


def title(fig, text, sub=None):
    fig.text(0.06, 0.93, text, fontsize=17, weight="semibold", color=INK, va="baseline")
    if sub:
        fig.text(0.06, 0.885, sub, fontsize=11, color=INK2, va="baseline")


def plain_rule(template):
    """The root line of a rule template in plain words."""
    text = template[-1].replace("W·", "").replace(" + b)", " + bias)").replace(" + b ", " + bias ")
    return text.replace("(x0)", "(a)").replace("(x1)", "(b)").replace("(x2)", "(=)").replace("Σ_unit", "Σ units")


def plain_labels(labels):
    """`Plane{5} Unit{3,17}` as `frequency 5 · 2 units`."""
    words = []
    for part in labels.split():
        kind, _, rest = part.partition("{")
        indices = rest.rstrip("}")
        count = sum(int(r.split("..")[1]) - int(r.split("..")[0]) + 1 if ".." in r else 1 for r in indices.split(",")) if indices else 1
        if kind == "Plane":
            words.append(("frequency " if count == 1 else "frequencies ") + indices)
        elif kind == "Unit":
            words.append(f"{count} unit" + ("s" if count > 1 else ""))
        elif kind == "Token":
            words.append(f"{count} token" + ("s" if count > 1 else ""))
    return " · ".join(words) or labels


def rules_figure(rung, path):
    rules = [r for r in rung["printout"]["rules"] if r["bits"] > 0]
    rows = []
    for index, rule in enumerate(rules):
        for instance in rule["instances"]:
            rows.append((index, instance))
    height = 1.6 + 0.32 * len(rows) + 0.5 * len(rules)
    fig = plt.figure(figsize=(12, max(4.5, height)), dpi=200)
    title(fig, "The model, read as a few rules",
          f"{len(rules)} rules, {len(rows)} uses; each bar is one use, its length the bits it costs")
    ax = fig.add_axes([0.42, 0.06, 0.52, 0.76])
    y, ticks, labels, colors = 0.0, [], [], [BLUE, ORANGE, AQUA]
    for index, rule in enumerate(rules):
        fig_y = None
        for instance in rule["instances"]:
            ax.barh(y, instance["bits"], height=0.72, color=colors[min(index, 2)], edgecolor=SURFACE, linewidth=2)
            ticks.append(y)
            labels.append(plain_labels(instance["labels"]))
            fig_y = y if fig_y is None else fig_y
            y += 1
        ax.text(-0.02, fig_y - 0.9, f"rule {index + 1}:  {plain_rule(rule['template'])}", transform=ax.get_yaxis_transform(),
                ha="right", fontsize=10.5, weight="semibold", color=INK)
        y += 1.2
    ax.set_yticks(ticks, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("bits")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def bits_figure(rungs, path):
    native = rungs[0]["printout"]["bits"]
    ladder = sorted(rungs[1:], key=lambda r: r["observations"])
    n = np.array([r["observations"] for r in ladder], dtype=float)
    algorithm = np.array([r["printout"]["bits"]["algorithm"] for r in ladder], dtype=float)
    constants = np.array([r["printout"]["bits"]["constants"] for r in ladder], dtype=float)
    data = np.array([r["printout"]["bits"]["data"] for r in ladder], dtype=float)
    fig = plt.figure(figsize=(10, 6), dpi=200)
    title(fig, "A short program that explains more and more of the model",
          "each point is the shortest program found when every input is observed n times")
    ax = fig.add_axes([0.1, 0.12, 0.62, 0.7])
    for values, color, label in [(algorithm, BLUE, "program: structure"), (constants, ORANGE, "program: numbers"),
                                 (np.maximum(data, 1e-3), AQUA, "behaviour not yet explained")]:
        ax.plot(n, values, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(n[-1] * 1.6, values[-1], label, color=INK, fontsize=10.5, va="center")
    native_bits = native["algorithm"] + native["constants"]
    ax.axhline(native_bits, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(n[0], native_bits * 1.25, f"the trained weights as stored: {native_bits:,} bits", color=INK2, fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("observations of each input, n")
    ax.set_ylabel("bits")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def unresolved_figure(rung, p, path):
    behaviour = rung["printout"]["unresolved"]["behaviour"]
    a = np.array(behaviour["tokens"][0])
    b = np.array(behaviour["tokens"][1])
    grid = np.zeros((p, p))
    grid[a, b] = behaviour["row_bits"]
    fig = plt.figure(figsize=(8.6, 8.2), dpi=200)
    total = behaviour["data_bits"]
    title(fig, "Where the program is not yet the model",
          f"bits of behaviour still unexplained for each input a + b, {total:,.1f} in all (n = {rung['observations']:,})")
    ax = fig.add_axes([0.1, 0.08, 0.72, 0.75])
    cmap = LinearSegmentedColormap.from_list("blue", SEQUENTIAL)
    image = ax.imshow(grid, cmap=cmap, origin="lower", interpolation="nearest")
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ticks = list(range(0, p, 5))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, cax=fig.add_axes([0.85, 0.2, 0.02, 0.5]))
    bar.outline.set_visible(False)
    bar.set_label("bits")
    fig.savefig(path)
    plt.close(fig)


def main():
    source, out, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    report = json.load(open(source))
    rungs = report["rungs"]
    p = int(report["export"]["output"]["n_classes"])
    os.makedirs(out, exist_ok=True)
    # The program chosen for the most behaviour.
    last = max(rungs[1:], key=lambda r: r["observations"])
    paths = [os.path.join(out, f"{prefix}_{name}.png") for name in ("rules", "bits", "unresolved")]
    rules_figure(last, paths[0])
    bits_figure(rungs, paths[1])
    unresolved_figure(last, p, paths[2])
    print("\n".join(paths))


if __name__ == "__main__":
    main()
