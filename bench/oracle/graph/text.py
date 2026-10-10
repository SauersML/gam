"""Real-text tasks for the graph oracle (#2951): explain how the model computes its next-token prediction after a text.

A task is a Pile validation sequence of the model's context length (512 tokens) cut after a position drawn uniformly,
so every prediction the model makes is equally likely to be asked about; the prediction after the last token is to be
explained. Train and held-out tasks come from disjoint rows.

The hard split asks about predictions that are not what the text suggests: in each row, the position where the
model's most probable next token differs from the text's actual next token and the model is most sure of it (its
probability highest). The text points to the actual token, so explaining why the model predicts another one takes
its computation, not the text.

  text.py build --split train --n 2000 --offset 6000     writes ~/mpd-data/graph_oracle/texts/vpd4l/<id>.json (a
  text.py build --split heldout --n 200 --offset 9000    behavior: one prompt, its target, the model's top next tokens)
  text.py build --split hard --n 50 --offset 9300        hard<row>.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from functools import lru_cache
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "vpd_2951"))

import mech  # noqa: E402

TEXTS = Path.home() / "mpd-data/graph_oracle/texts"
CONTEXT = 512  # vpd4l's context length (model_config.yaml n_ctx)


@lru_cache(None)
def model(device: str = "mps"):
    """vpd4l."""
    import vpd_model

    return vpd_model.load_target(device)


def top(ids: list[int], k: int = 3, device: str = "mps") -> list[list]:
    """The model's k most probable next tokens after `ids`, with their probabilities."""
    with torch.no_grad():
        p = model(device)(torch.tensor([ids], device=device))[0, -1].softmax(-1)
    tk = mech.tokenizer("vpd4l")
    v, i = p.topk(k)
    return [[tk.decode([int(j)]), round(float(q), 4)] for q, j in zip(v.tolist(), i.tolist())]


def build(split: str, n: int, offset: int, seed: int, device: str = "mps") -> None:
    import vpd_model

    rng = random.Random(seed)
    rows = vpd_model.val_tokens(n, CONTEXT + 1, offset)  # the token after the context: the actual next token at the last position
    (TEXTS / "vpd4l").mkdir(parents=True, exist_ok=True)
    tk = mech.tokenizer("vpd4l")
    for r in range(n):
        if split == "hard":
            with torch.no_grad():
                p = model(device)(rows[r:r + 1, :CONTEXT].to(device))[0].float().softmax(-1)
            sure, guess = p.max(-1)
            sure[guess == rows[r, 1:CONTEXT + 1].to(device)] = -1.0  # right predictions are not asked about
            t = int(sure.argmax())
        else:
            t = rng.randrange(CONTEXT)
        ids = rows[r, :t + 1].tolist()
        name = f"{'hard' if split == 'hard' else 'text'}{offset + r}"
        prompt = {"text": tk.decode(ids), "token_ids": ids, "target_positions": [t], "model_top": [top(ids, device=device)],
                  "actual_next": tk.decode([int(rows[r, t + 1])])}
        (TEXTS / "vpd4l" / f"{name}.json").write_text(json.dumps({"id": name, "model": "vpd4l", "family": "text", "split": split,
                                                                  "description": "", "prompts": [prompt]}))
        if (r + 1) % 100 == 0:
            print(f"{split}: {r + 1}/{n}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split", choices=("train", "heldout", "hard"), required=True)
    b.add_argument("--n", type=int, required=True)
    b.add_argument("--offset", type=int, required=True, help="the first Pile validation row (train and heldout rows must not overlap)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--device", default="mps")
    a = ap.parse_args()
    build(a.split, a.n, a.offset, a.seed, a.device)


if __name__ == "__main__":
    main()
