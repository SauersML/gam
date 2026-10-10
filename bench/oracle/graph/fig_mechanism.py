"""An answer's mechanism as a picture (#2951 graph oracle): its first steps' subcomponents placed by the text position
they act at (across) and the layer and weight matrix they belong to (up), the connections between them as lines,
earlier steps darker; beside it, the KL of the model's prediction from the graph of the first k steps, measured as
the verifier measures it (native.faithfulness on the text's changed prompts).

  fig_mechanism.py TASK_ID ANSWER.py OUT.png [--steps K]
  fig_mechanism.py TASK_ID EVAL_SAMPLES.jsonl OUT.png --program prompted_r3   (that program's best answer to TASK_ID)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402
import native  # noqa: E402

ROWS = [("q_proj", "k_proj", "v_proj"), ("o_proj",), ("c_fc",), ("down_proj",)]  # bottom to top within a layer
ROW_NAMES = ("attention in", "attention out", "MLP in", "MLP out")
MARKERS = {"q_proj": "v", "k_proj": "^", "v_proj": "D", "o_proj": "o", "c_fc": "s", "down_proj": "o"}


def answer_source(path: Path, task_id: str, program: str | None) -> str:
    if path.suffix == ".py":
        return path.read_text()
    import score

    rows = [r for r in map(json.loads, open(path)) if r["behavior"] == task_id and r["program"].startswith(program)
            and r["score"].get("valid", True) and r["score"].get("curve")]
    return min(rows, key=lambda r: score.key(r["score"]))["source"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task")
    ap.add_argument("answer", type=Path)
    ap.add_argument("out")
    ap.add_argument("--program", default="prompted")
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()
    task = json.loads((native.TEXTS / "vpd4l" / f"{a.task}.json").read_text())
    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    t0 = targets[0]
    strings = mech.behavior_tokens(task, "vpd4l")["sequences"][0][1]
    ir = mech.trace_inline(answer_source(a.answer, a.task, a.program), "vpd4l", task)
    assert ir["valid"], ir["error"]
    g = ir["graph"]
    n_steps = g["steps"]
    k = min(a.steps, n_steps)
    nat = native.Native()
    graphs = [native.prefix(ir, j)[0] for j in range(1, n_steps + 1)]
    changed = nat.changes(ids, targets, seed=native.task_seed(a.task))
    kls = nat.faithfulness(ids, targets, [native.Graph()] + graphs, changed, seed=native.task_seed(a.task))

    keep = [i for i, s in enumerate(g["node_step"]) if s < k]
    positions = sorted({g["nodes"][i][2] for i in keep} | {t0})
    slot = {p: j for j, p in enumerate(positions)}
    colors = cm.Blues_r([0.05 + 0.6 * j / max(k - 1, 1) for j in range(k)])  # step 1 darkest

    def row_of(kind):
        return next(r for r, kinds in enumerate(ROWS) if kind in kinds)

    # several subcomponents at one (row, position): spread them across the position's slot
    groups = {}
    for i in keep:
        layer, kind, pos, c = g["nodes"][i]
        groups.setdefault((layer, row_of(kind), pos), []).append(i)
    xy = {}
    for (layer, row, pos), members in groups.items():
        for m, i in enumerate(sorted(members, key=lambda i: g["node_step"][i])):
            spread = 0.7 * ((m + 0.5) / len(members) - 0.5) if len(members) > 1 else 0.0
            xy[i] = (slot[pos] + spread, layer * len(ROWS) + row)
    top_y = 4 * len(ROWS)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(17, 8.5), gridspec_kw={"width_ratios": [3.2, 1]})
    for (r, w), s in zip(g["parents"], g["parent_step"]):
        if s < k and r in xy and w in xy:
            ax.plot([xy[w][0], xy[r][0]], [xy[w][1], xy[r][1]], color=colors[s], lw=1.2, alpha=0.55, zorder=1)
    for w, s in zip(g["out"], g["out_step"]):
        if s < k and w in xy:
            ax.plot([xy[w][0], slot[t0]], [xy[w][1], top_y], color=colors[s], lw=1.2, alpha=0.55, zorder=1)
    for i in sorted(keep, key=lambda i: -g["node_step"][i]):
        layer, kind, pos, c = g["nodes"][i]
        ax.scatter(*xy[i], s=46, marker=MARKERS[kind], color=colors[g["node_step"][i]], edgecolor="white", linewidth=0.6, zorder=2)
    ax.scatter([slot[t0]], [top_y], s=260, marker="*", color="#b03a2e", zorder=3)
    guess, actual = prompt["model_top"][0][0][0], prompt.get("actual_next")
    label = f"predicts {guess!r}" + (f" (the text has {actual!r})" if actual is not None and actual != guess else "")
    ax.annotate(label, (slot[t0], top_y), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=15, color="#b03a2e")
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"{strings[p]!r}\n{p}" for p in positions], fontsize=12, rotation=0)
    ax.set_yticks([layer * len(ROWS) + r for layer in range(4) for r in range(len(ROWS))])
    ax.set_yticklabels([f"layer {layer} {ROW_NAMES[r]}" for layer in range(4) for r in range(len(ROWS))], fontsize=12)
    ax.set_xlim(-0.7, len(positions) - 0.3)
    ax.set_ylim(-0.8, top_y + 1.6)
    ax.set_xlabel("text position", fontsize=15)

    steps = list(range(0, n_steps + 1))
    bx.plot(steps, kls, color="#7f7f7f", lw=2, zorder=1)
    for j in range(k):
        bx.scatter([j + 1], [kls[j + 1]], s=70, color=colors[j], zorder=2)
    bx.scatter([0], [kls[0]], s=70, color="#7f7f7f", zorder=2)
    bx.set_yscale("log")
    bx.set_xlabel("steps of the answer", fontsize=15)
    bx.set_ylabel("KL of the prediction (bits)", fontsize=15)
    for axis in (ax, bx):
        axis.tick_params(labelsize=12)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130, facecolor="white")
    print(f"{a.out}: {a.task}, first {k} of {n_steps} steps, KL {kls[0]:.1f} -> " + " -> ".join(f"{x:.2f}" for x in kls[1:k + 1]) + f" ... {kls[-1]:.2f}")


if __name__ == "__main__":
    main()
