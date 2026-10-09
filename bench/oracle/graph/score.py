"""Scoring graph answers (#2951 graph oracle): the verifier's numbers for each answer.

An answer (mech.py) is traced into ordered steps. The graph of its first k steps runs alone on the model (native.py)
over the text's changed prompts, giving its faithfulness (KL in bits) and its description length (bits); with the empty
graph at 0 bits these points are the answer's curve. One curve is better than another at a description length when its
best point within that length has the lower KL. An answer's score (key) is area(): the mean KL its curve reaches within a
description length drawn log-uniformly from one subcomponent's ("lo") to the whole model's, every subcomponent at every
position ("hi"): from the smallest explanation there is to the model itself, and every doubling of length counts the
same (there is no natural unit of explanation size). The range is the question's alone, so scores compare across
answers, groups and baselines. No tolerance and no weight: an answer that explains nothing keeps the empty graph's KL
over the whole range, the worst curve there is.

  score.py ANSWER.py TASK.json      prints the answer's score
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EMPTY = 'def graph(tokens, targets):\n    return []\n'  # the graph with no nodes: the embedding alone
LIBRARY = Path.home() / "mpd-data/graph_oracle/texts/library.json"  # mined recurring mechanisms (mine.py), if any


def area(curve: list, lo: float, hi: float) -> float:
    """The mean over description lengths s log-uniform in [lo, hi] of the lowest KL among the curve's points
    [bits, kl] with bits <= s (the empty graph's point [0, kl] is always among them)."""
    if hi <= lo:
        return min(k for b, k in curve if b <= lo)
    pts = sorted((max(b, lo), k) for b, k in curve if b <= hi)
    total, cur, at = 0.0, math.inf, math.log(lo)
    for b, k in pts:
        x = math.log(b)
        if x > at:
            total += cur * (x - at)
            at = x
        cur = min(cur, k)
    total += cur * (math.log(hi) - at)
    return total / (math.log(hi) - math.log(lo))


def key(s: dict) -> tuple:
    """A score's rank, lower first: invalid last, then area() over the question's range."""
    if not s.get("valid", True) or not s.get("curve"):
        return (1, math.inf)
    return (0, area(s["curve"], s["lo"], s["hi"]))


def keys(scores: list[dict]) -> list[tuple]:
    return [key(s) for s in scores]


class Scorer:
    """Traces and runs answers on one device."""

    def __init__(self, dev: str | None = None, library: Path | None = LIBRARY):
        import native

        self.native = native
        self.nat = native.Native(dev)
        self.library = json.loads(Path(library).read_text())["entries"] if library and Path(library).exists() else {}

    def expand(self, g, uses: list[str], targets: list[int], T: int):
        """g with the library entries it uses added (their edges at the target's offsets); None when an entry is
        unknown, falls outside the sequence or is not a connection the model has."""
        import mech

        t0 = targets[0]
        for name in uses:
            entry = self.library.get(name)
            if entry is None:
                return None
            for w_tok, wo, r_tok, ro in entry["edges"]:
                m = mech.PART.fullmatch(w_tok)
                w = (self.native.site_name(int(m[1]), mech.SITES[m[2]]), t0 + wo, int(m[3]))
                if not 0 <= w[1] < T:
                    return None
                if r_tok == "out":
                    if not self.native.connects(w, None, targets):
                        return None
                    if w not in g.out:
                        g.out.append(w)
                    g.nodes.add(w)
                    continue
                m = mech.PART.fullmatch(r_tok)
                r = (self.native.site_name(int(m[1]), mech.SITES[m[2]]), t0 + ro, int(m[3]))
                if not 0 <= r[1] < T or not self.native.connects(w, r, targets):
                    return None
                if w not in g.parents.setdefault(r, []):
                    g.parents[r].append(w)
                g.nodes |= {w, r}
        return g

    def score(self, task: dict, sources: list[str], seed: int = 0, interchange: bool = False) -> list[dict]:
        """Each answer's score on a task (a text task record) under one draw of changed prompts (seed): {"valid",
        "error", "curve": [[bits, kl], ...] (the empty graph, then each step), "lo" and "hi": one subcomponent's and
        the whole model's description lengths, "kl_bits" and "bits" (the whole answer), "steps", "nodes", "edges",
        "explanation", "notes", "interchange_kl_bits" (with interchange: native.interchange of the whole answer, its
        steps as the groups; an evaluation measure), "dropped": what the answer wrote that is not part of its graph (mech; its description
        length still counts), "events": the changed prompts' token changes and whether each flips the model's top next
        token (native.flips, the same for every answer)}; the
        source "vpd" stands for VPD's own answer (complete, one step)."""
        import mech

        prompt = task["prompts"][0]
        ids, targets = prompt["token_ids"], prompt["target_positions"]
        nat = self.nat
        prompts = nat.changes(ids, targets, seed=(self.native.task_seed(task["id"]) + 1_000_003 * seed) % (1 << 31))
        positions, total = nat.positions(targets), sum(nat.C.values())
        ref_bits = math.log2(max(len(self.library), 1))  # a library reference: one choice among the entries
        graphs, owners, own_bits = [self.native.Graph()], [None], [0.0]  # every graph to run, which answer it belongs to, its description length
        out = []
        for j, src in enumerate(sources):
            if src == "vpd":
                g = nat.vpd_answer(ids)
                graphs.append(g)
                owners.append(j)
                own_bits.append(g.bits(positions, total))
                out.append({"valid": True, "error": None, "steps": 1, "explanation": "", "notes": []})
                continue
            ir = mech.trace(src, "vpd4l", behavior=task)
            out.append({"valid": ir["valid"], "error": ir.get("error"), "steps": ir["graph"].get("steps", 0), "explanation": ir["graph"].get("explanation", ""),
                        "notes": ir["graph"].get("notes", [])})
            if not ir["valid"]:
                continue
            mine = []
            dropped = ir["graph"].get("dropped", [])
            out[-1]["dropped"] = [why for _, why in dropped]
            for k in range(1, out[-1]["steps"] + 1):
                g, uses = self.native.prefix(ir, k)
                written = sum(1 for st, _ in dropped if st < k)  # written, not in the graph: each costs a subcomponent's description
                bits = g.bits(positions, total) + ref_bits * len(uses) + written * math.log2(positions * total)
                if uses:
                    g = self.expand(g, uses, targets, len(ids))
                    if g is None:
                        out[-1].update(valid=False, error=f"uses: an entry of {uses} is unknown, outside the text or not a model connection")
                        break
                mine.append((g, bits))
            if out[-1]["valid"]:
                for g, bits in mine:
                    graphs.append(g)
                    owners.append(j)
                    own_bits.append(bits)
                if interchange and mine:
                    whole = mine[-1][0]
                    steps = [self.native.prefix(ir, k)[0].node_set() for k in range(1, out[-1]["steps"] + 1)]
                    groups = [b - a for a, b in zip([set()] + steps[:-1], steps) if b - a]
                    out[-1]["interchange_kl_bits"] = nat.interchange(ids, targets, whole, groups, prompts, seed)
        kl = nat.faithfulness(ids, targets, graphs, prompts)
        lo = math.log2(positions * total)
        events = nat.flips(ids, targets, prompts)  # what the English reader is asked about (reader.py)
        for s in out:
            s.update(curve=[[0.0, kl[0]]], lo=lo, hi=positions * total * lo, events=events)
        for g, o, b, k in zip(graphs[1:], owners[1:], own_bits[1:], kl[1:]):
            s = out[o]
            s["curve"].append([b, k])
            s.update(kl_bits=k, bits=b, nodes=g.count(), edges=None if g.complete else g.edges())
        for s in out:
            if s["valid"] and len(s["curve"]) == 1:  # no steps: the empty graph
                s.update(kl_bits=kl[0], bits=0.0, nodes=0, edges=0)
        return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("answer", type=Path)
    ap.add_argument("task", type=Path)
    a = ap.parse_args()
    s = Scorer().score(json.loads(a.task.read_text()), [a.answer.read_text()])[0]
    print(json.dumps({**s, "area": key(s)[1]}))


if __name__ == "__main__":
    main()
