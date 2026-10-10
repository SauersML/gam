"""What a prediction reads from the text, shown by changing the text (#2951 graph oracle): the model's probabilities of
a few tokens after the text as it is, after one earlier token is replaced, and after only the last tokens are kept.
On hard9308 (the model predicts "67" after "#define MIN_HWORD (-327" where the text has "68") replacing the earlier
"67" of "32767" moves the prediction with it, and the last tokens alone give "68": the model copies.

  fig_counterfactual.py OUT.png [--task hard9308 --position 322 --replace 12 --last 6 --tokens 67,68,12]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402
import native  # noqa: E402
import text  # noqa: E402

COLORS = ["#2a78b5", "#d98a1e", "#1d8a6b", "#8a5cc4"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--task", default="hard9308")
    ap.add_argument("--position", type=int, default=322)
    ap.add_argument("--replace", default="12", help="the token put at --position")
    ap.add_argument("--last", type=int, default=6, help="tokens kept in the third condition")
    ap.add_argument("--tokens", default="67,68,12", help="the next tokens whose probabilities are drawn")
    ap.add_argument("--device", default=native.device())
    a = ap.parse_args()
    task = json.loads((native.TEXTS / "vpd4l" / f"{a.task}.json").read_text())
    ids = task["prompts"][0]["token_ids"]
    tk = mech.tokenizer("vpd4l")
    model = text.model(a.device)
    shown = a.tokens.split(",")
    tid = [tk.token_to_id(x) for x in shown]
    edited = list(ids)
    old = tk.decode([ids[a.position]])
    edited[a.position] = tk.token_to_id(a.replace)
    conditions = [("the text", ids), (f"earlier {old!r} -> {a.replace!r}", edited), (f"last {a.last} tokens only", ids[-a.last:])]
    probs = []
    with torch.no_grad():
        for _, x in conditions:
            p = model(torch.tensor([x], device=a.device))[0, -1].float().softmax(-1)
            probs.append([float(p[i]) for i in tid])
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.8 / len(shown)
    for j, tok in enumerate(shown):
        xs = [i + (j - (len(shown) - 1) / 2) * width for i in range(len(conditions))]
        ax.bar(xs, [p[j] for p in probs], width * 0.92, color=COLORS[j % len(COLORS)], label=repr(tok))
        for x, p in zip(xs, probs):
            if p[j] >= 0.02:
                ax.annotate(f"{p[j]:.2f}", (x, p[j]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=13)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c for c, _ in conditions], fontsize=14)
    ax.set_ylabel("model's probability of the next token", fontsize=15)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=14, frameon=False, title="next token", title_fontsize=13)
    ax.tick_params(labelsize=13)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130, facecolor="white")
    for (c, _), p in zip(conditions, probs):
        print(f"{c}: " + ", ".join(f"{t!r} {q:.3f}" for t, q in zip(shown, p)))


if __name__ == "__main__":
    main()
