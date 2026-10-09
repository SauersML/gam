"""Scoring graph answers (#2951 graph oracle): an answer is traced (mech.py) into a graph of subcomponents at positions
and run on the model (native.py), which reports its KL under deletion and random ablation of what it leaves out
(kl_bits, the larger) and its size (nodes plus edges). Each interchange claim (position p, replacement, top) is checked
on the model: replacing token p changes the top prediction to `top`, and the graph's pathway from p alone (its
activations from the edited run, everything else from the original) gives `top` too (native.interchange).

order(score, eps) ranks answers to one question: valid first, then KL above eps (none is best), then fewer wrong
claims, then more source positions with a correct claim, then fewer nodes plus edges. No weight trades one for another:
eps, the precision the question asks for, decides what is faithful.

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
    """How a score ranks, lower first: valid, then its KL above eps, then its size (a complete graph, which declares no
    edges, as large as can be)."""
    if not s.get("valid", True):
        return (1, math.inf, math.inf, 0, math.inf)
    return (0, max(0.0, s["kl_bits"] - eps), s.get("claims_wrong", 0), -s.get("claims_explained", 0),
            s["size"] if s.get("size") is not None else math.inf)


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
        out, graphs, where, claims = [], [], [], []  # graphs, their places in out and their claims, aligned
        for src in sources:
            if src == "vpd":
                graphs.append(self.nat.vpd_answer(ids))
                where.append(len(out))
                claims.append([])
                out.append({"valid": True, "error": None})
                continue
            ir = mech.trace(src, "vpd4l", behavior=task)
            out.append({"valid": ir["valid"], "error": ir.get("error")})
            if ir["valid"]:
                graphs.append(self.nat.from_ir(ir))
                where.append(len(out) - 1)
                claims.append(ir["graph"].get("claims", []))
        for j, s in zip(where, self.nat.score(ids, targets, graphs, seed)):
            out[j].update(s)
        for j, g, cs in zip(where, graphs, claims):
            out[j].update(self.check_claims(ids, targets, g, cs))
        return out

    def check_claims(self, ids: list[int], targets: list[int], g, claims: list) -> dict:
        """{"claims", "claims_wrong", "claims_explained"}: how many claims, how many fail, and how many source
        positions have a claim that holds."""
        import mech

        tk = mech.tokenizer("vpd4l")
        paths = self.native.pathways(g, targets)
        good = set()
        wrong = 0
        for p, rep, top in claims:
            rep_ids, top_ids = tk.encode(rep, add_special_tokens=False).ids, tk.encode(top, add_special_tokens=False).ids
            nodes = paths.get(p)
            if not nodes or len(rep_ids) != 1 or len(top_ids) != 1:  # a claim must name one token each and a pathway
                wrong += 1
                continue
            rep_id, top = rep_ids[0], top_ids[0]
            r = self.nat.interchange(ids, [ids[:p] + [rep_id] + ids[p + 1:]], targets, nodes)[0]
            if r["edited_top"][0] == top and r["patched_top"][0] == top and r["top"][0] != top:
                good.add(p)
            else:
                wrong += 1
        return {"claims": len(claims), "claims_wrong": wrong, "claims_explained": len(good)}


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
