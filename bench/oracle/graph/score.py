"""Scoring graph answers (#2951 graph oracle): an answer is traced (mech.py) into a graph of subcomponents at positions
and run on the model (native.py), which reports its KL under deletion and random ablation of what it leaves out
(kl_bits, the larger) and its size (nodes plus edges).

order(score, eps) ranks answers to one question: valid first, then KL above eps (none is best), then fewer nodes plus
edges. No weight trades KL against size: eps, the precision the question asks for, decides what is correct.

  score.py ANSWER.py TASK.json [--eps 0.5]      prints the answer's score
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EMPTY = 'def graph(tokens, targets):\n    return {"out": ""}\n'  # the graph with no nodes: the embedding alone


def order(s: dict, eps: float) -> tuple:
    """How a score ranks, lower first: valid, then its KL above eps, then its size."""
    if not s.get("valid", True):
        return (1, math.inf, math.inf)
    return (0, max(0.0, s["kl_bits"] - eps), s["size"])


class Scorer:
    """Traces and runs answers on one device."""

    def __init__(self, dev: str | None = None):
        import native

        self.native = native
        self.nat = native.Native(dev)

    def score(self, task: dict, sources: list[str], seed: int = 0) -> list[dict]:
        """Each answer's score on a task (a text task record): {"valid", "error", "kl_bits", "kl_deleted_bits",
        "kl_random_bits", "nodes", "edges", "size"}; the source "vpd" stands for VPD's own answer (complete)."""
        import mech

        prompt = task["prompts"][0]
        ids, targets = prompt["token_ids"], prompt["target_positions"]
        out, graphs, where = [], [], []
        for src in sources:
            if src == "vpd":
                graphs.append(self.nat.vpd_answer(ids))
                where.append(len(out))
                out.append({"valid": True, "error": None})
                continue
            ir = mech.trace(src, "vpd4l", behavior=task)
            out.append({"valid": ir["valid"], "error": ir.get("error")})
            if ir["valid"]:
                graphs.append(self.nat.from_ir(ir))
                where.append(len(out) - 1)
        for j, s in zip(where, self.nat.score(ids, targets, graphs, seed)):
            out[j].update(s)
        return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("answer", type=Path)
    ap.add_argument("task", type=Path)
    ap.add_argument("--eps", type=float, default=0.5)
    a = ap.parse_args()
    s = Scorer().score(json.loads(a.task.read_text()), [a.answer.read_text()])[0]
    print(json.dumps({**s, "order": order(s, a.eps)}))


if __name__ == "__main__":
    main()
