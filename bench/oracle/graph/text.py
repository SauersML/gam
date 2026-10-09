"""Real-text tasks for the graph oracle (#2951): explain the model's next-token prediction at the end of a window of
real text by the circuit that computes it.

A task is a window of LENGTH tokens of Pile validation text; the prediction after its last token is to be explained.
An explanation is a gate program (mech.gates): which subcomponents act at which positions. With every subcomponent it
leaves out deleted (VPD's ablation), it is judged by the KL in bits of the model's next-token distribution there from
the circuit's, against VPD's own answer on the same text: at every position, the subcomponents whose causal importance
there is above zero (VPD's definition of a needed subcomponent, its ablation as published). An explanation as faithful
as VPD's (a KL no larger) is better the fewer (subcomponent, position) pairs it names; one less faithful is worse by its
excess (score.order). VPD's answer is the teacher and each task's reference: on 20 held-out windows it keeps 0.44 bits
with 7,090 pairs (its importance at the predicted position alone: 1.11 bits with 182). A position's subcomponents are
written as one string of names, one oracle token each.

  text.py build --split train --n 2000 --offset 6000    writes ~/mpd-data/graph_oracle/texts/vpd4l/<id>.json (a
  text.py build --split heldout --n 200 --offset 9000   behavior: one prompt, its target, the model's top next tokens)
and texts/teacher/<id>.py (train) or texts/teacher_heldout/<id>.py (heldout, the evaluation's VPD baseline).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "vpd_2951"))

import atlas  # noqa: E402
import mech  # noqa: E402

TEXTS = Path.home() / "mpd-data/graph_oracle/texts"
LENGTH = 32


def teacher(ids: list[int], device: str = "mps") -> str:
    """VPD's answer: at every position, the subcomponents whose causal importance there is above zero (by layer, site
    and index)."""
    per: dict[int, list[tuple]] = {}
    for site, m in atlas.importance(ids, device).items():
        layer, kind = atlas.site_of(site)
        for t, i in (m > 0).nonzero().tolist():
            per.setdefault(t, []).append((layer, list(mech.SITES.values()).index(kind), i, atlas.part(layer, kind, i)))
    lines = [f'        {t}: "' + "".join(p for *_, p in sorted(per[t])) + '",' for t in sorted(per)]
    return "def on(tokens, targets):\n    return {\n" + "\n".join(lines) + "\n    }\n"


def top(ids: list[int], k: int = 3, device: str = "mps") -> list[list]:
    """The model's k most probable next tokens after `ids`, with their probabilities."""
    target, _ = atlas.model(device)
    with torch.no_grad():
        p = target(torch.tensor([ids], device=device))[0, -1].softmax(-1)
    tk = mech.tokenizer("vpd4l")
    v, i = p.topk(k)
    return [[tk.decode([int(j)]), round(float(q), 4)] for q, j in zip(v.tolist(), i.tolist())]


def build(split: str, n: int, offset: int, seed: int, device: str = "mps") -> None:
    import vpd_model

    rng = random.Random(seed)
    rows = vpd_model.val_tokens(n, 512, offset)
    (TEXTS / "vpd4l").mkdir(parents=True, exist_ok=True)
    answers = TEXTS / ("teacher" if split == "train" else "teacher_heldout")
    answers.mkdir(exist_ok=True)
    tk = mech.tokenizer("vpd4l")
    for r in range(n):
        start = rng.randrange(0, rows.shape[1] - LENGTH)
        ids = rows[r, start:start + LENGTH].tolist()
        name = f"text{offset + r}"
        prompt = {"text": tk.decode(ids), "token_ids": ids, "target_positions": [LENGTH - 1], "model_top": [top(ids, device=device)]}
        (TEXTS / "vpd4l" / f"{name}.json").write_text(json.dumps({"id": name, "model": "vpd4l", "family": "text", "split": split,
                                                                  "description": "", "prompts": [prompt]}))
        (answers / f"{name}.py").write_text(teacher(ids, device))
        if (r + 1) % 100 == 0:
            print(f"{split}: {r + 1}/{n}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split", choices=("train", "heldout"), required=True)
    b.add_argument("--n", type=int, required=True)
    b.add_argument("--offset", type=int, required=True, help="the first Pile validation row (train and heldout rows must not overlap)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--device", default="mps")
    a = ap.parse_args()
    build(a.split, a.n, a.offset, a.seed, a.device)


if __name__ == "__main__":
    main()
