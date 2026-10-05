"""Wiring graph of a library read-out's most-used functions (mpd_library_readout_2951 JSON).

python library_readout_wiring.py READOUT.json OUT.png EDGES

The drawn functions are the read-out's core: the largest RelP importance (mean |attribution| of
the model's predicted-token logit over held-out tokens), functions active on every token apart.
Columns are the reads in depth order (layer l's heads, then its MLP functions); a node's area is
its importance. An MLP function is labelled with the token it fires on most often among its top
held-out contexts; a head with the diagnostic holding most of its attention (previous token,
induction, duplicate token), else its median query-key offset, and "copies" when most source
tokens' largest OV output is the token itself. The second line is the predicted token a majority
of the function's largest attributions support, else "no consistent output token". Arrows are
RelP flow edges: the writer's direct effect on the reader's reads (an MLP's gate and up maps, a
head's query, key and value maps) times the reader's gradient of m, summed over held-out tokens;
each function shows its EDGES strongest inputs from the other drawn functions, an arrow's width is
its |flow| relative to the largest drawn, blue where the flow raises m and orange where it lowers m.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def shown(text):
    return "«" + text.replace("\n", "\\n").replace(" ", "␣") + "»"


def labels(f):
    output = f"supports {shown(f['supports_token_text'])}" if f.get("supports_token_text") is not None else "no consistent output token"
    if f["kind"] == "head":
        a = f["attention"]
        scores = {"previous-token": a["previous"], "induction": a["induction"], "duplicate-token": a["duplicate"]}
        name, score = max(scores.items(), key=lambda kv: kv[1])
        if score > 0.5:
            what = f"{name} head"
        else:
            total, running, median = sum(a["offset_mass"]), 0.0, a["offsets"][-1]
            for (lo, hi), m in zip(a["offsets"], a["offset_mass"]):
                running += m
                if running >= total / 2:
                    median = (lo, hi)
                    break
            lo, hi = median
            what = f"median offset {lo}" if lo == hi else f"median offset {lo}–{hi}"
        if a["self_top1"] > 0.5:
            what += ", copies"
        return what, output
    tokens = [c["token"] for c in f["contexts"]]
    fires = max(tokens, key=tokens.count) if tokens else ""
    return f"fires on {shown(fires)}", output


def main():
    path, out, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
    report = json.load(open(path))
    functions = report["functions"]
    core = report["core"]
    wiring = report["wiring"]
    column = {i: 2 * functions[i]["layer"] + (functions[i]["kind"] == "mlp") for i in core}
    columns = sorted(set(column.values()))
    x_of = {c: k for k, c in enumerate(columns)}
    stacks = {c: [i for i in core if column[i] == c] for c in columns}
    position = {}
    for c, members in stacks.items():
        for r, i in enumerate(members):
            position[i] = (x_of[c] * 4.0, -r * 1.25)
    fig, ax = plt.subplots(figsize=(5.0 * len(columns), 1.3 * max(len(m) for m in stacks.values()) + 1.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    edges = []
    for a in range(len(core)):
        inputs = sorted(((abs(wiring[a][b]), wiring[a][b] > 0, core[b], core[a]) for b in range(len(core)) if wiring[a][b] != 0), reverse=True)
        edges.extend(inputs[:count])
    top = max((w for w, _, _, _ in edges), default=1.0)
    for w, raises, writer, reader in sorted(edges):
        ax.add_patch(
            FancyArrowPatch(
                position[writer],
                position[reader],
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=0.4 + 6.0 * w / top,
                color="#3b6ea5" if raises else "#d9822b",
                alpha=0.25 + 0.6 * w / top,
                connectionstyle="arc3,rad=0.12",
                shrinkA=10,
                shrinkB=10,
            )
        )
    usage = max(functions[i]["importance"] for i in core)
    for i in core:
        f = functions[i]
        x, y = position[i]
        color = "#c0392b" if f["kind"] == "head" else "#2c3e50"
        ax.scatter([x], [y], s=120 + 600 * f["importance"] / usage, color=color, zorder=3)
        read, write = labels(f)
        ax.text(x + 0.22, y + 0.18, f["name"], fontsize=15, fontweight="bold", va="bottom", zorder=4)
        ax.text(x + 0.22, y + 0.12, f"{read}\n{write}", fontsize=13, va="top", zorder=4)
    for c in columns:
        layer, kind = divmod(c, 2)
        ax.text(x_of[c] * 4.0, 1.0, f"layer {layer} {'MLP' if kind else 'heads'}", fontsize=17, ha="left", va="bottom")
    ax.set_axis_off()
    ax.set_xlim(-0.6, 4.0 * len(columns) + 0.4)
    ax.set_ylim(-1.25 * max(len(m) for m in stacks.values()), 1.6)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
