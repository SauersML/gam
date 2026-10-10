"""Scoring graph answers (#2951 graph oracle): the verifier's numbers for each answer.

An answer (mech.py) is a program run on the text and on each changed prompt of it (a rule may bind differently there);
the graph of its first k steps is tested on the model (native.py) over experiments on the changed prompts, giving its
faithfulness (KL in bits), and its description length is its code's (code_bits, compressed; the English is scored by the
reader instead); with the empty
graph at 0 bits these points are the answer's curve. One curve is better than another at a description length when its
best point within that length has the lower KL. An answer's score (key) is area(): the mean KL its curve reaches within a
description length drawn log-uniformly from the empty program's ("lo") to the whole model's listing, every subcomponent
at every position ("hi"). The log-uniform weighting is a choice (no length scale preferred over another), a training
convenience; the curve itself is reported too. The range is the question's alone, so scores compare across answers,
groups and baselines. No tolerance: an answer that explains nothing keeps the empty graph's KL over the whole range,
the worst curve there is.

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


def strip(tree):
    """graph()'s docstring removed from a parsed answer (the English is scored by the reader, not counted as code)."""
    import ast

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "graph" and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    return tree


def prefix_code(source: str, k: int) -> str:
    """An answer's code (no docstring, no comments) with the list graph() returns cut to its first k entries when it
    returns a list written out; otherwise the whole code (its steps come from code that cannot be cut)."""
    import ast

    try:
        tree = strip(ast.parse(source))
    except SyntaxError:
        return source
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "graph":
            for ret in ast.walk(node):
                if isinstance(ret, ast.Return) and isinstance(ret.value, ast.List):
                    ret.value.elts = ret.value.elts[:k]
    return ast.unparse(tree)


def code_bits(source: str) -> float:
    """An answer's description length in bits: its code (no docstring, no comments, normalized by ast.unparse)
    compressed with LZMA2 in a raw stream. A rule that computes many connections costs its code, a list of
    connections its entries; a lookup table costs its table."""
    import ast
    import lzma

    try:
        code = ast.unparse(strip(ast.parse(source)))
    except SyntaxError:
        code = source
    return 8.0 * len(lzma.compress(code.encode(), format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]))


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

    def instances(self, src: str, task: dict, prompts: list) -> list:
        """The answer program's IR on each changed prompt (the task with that prompt's tokens): a rule may bind
        differently there. An invalid trace (the program fails on that text) is None."""
        import mech

        out = []
        for x in prompts:
            t = json.loads(json.dumps(task))
            t["prompts"][0]["token_ids"] = list(x)
            ir = mech.trace(src, "vpd4l", behavior=t)
            out.append(ir if ir["valid"] else None)
        return out

    def score(self, task: dict, sources: list[str], seed: int = 0, necessity: bool = False) -> list[dict]:
        """Each answer's score on a task (a text task record) under one draw of changed prompts (seed): {"valid",
        "error", "curve": [[bits, kl], ...] (the empty program, then each step), "lo" and "hi": the empty program's and
        the whole model's description lengths, "kl_bits" and "bits" (the whole answer), "steps", "nodes", "edges",
        "explanation", "notes", "necessity_kl_bits" (with necessity: native.necessity of the whole answer on the text,
        an evaluation measure), "dropped": what the answer wrote that is not part of its graph (mech), "events": the
        English reader's questions with the model's answers (native.events: every changed prompt, and holds of the
        answer's steps)}. An answer's description length is its code's
        (code_bits; library entries it uses are given, their definitions not counted); its first k steps are the same
        program with its returned list cut to k entries (prefix_code). The program runs on every changed prompt
        (instances), its graph there tested there. The source "vpd" stands for VPD's own answer (complete, one step;
        its description length the listing of its subcomponents, Graph.bits)."""
        import mech

        prompt = task["prompts"][0]
        ids, targets = prompt["token_ids"], prompt["target_positions"]
        nat = self.nat
        prompts = nat.changes(ids, targets, seed=(self.native.task_seed(task["id"]) + 1_000_003 * seed) % (1 << 31))
        positions, total = nat.positions(targets), sum(nat.C.values())
        lo = code_bits(EMPTY)
        P = len(prompts)
        rows = [(None, lo, [self.native.Graph()] * P)]  # (answer, description length, its graph on each prompt): the empty program first
        out, step_nodes = [], []  # step_nodes[j]: answer j's graph on the text after each step (node sets)
        for j, src in enumerate(sources):
            step_nodes.append([])
            if src == "vpd" and not nat.has_importance:
                out.append({"valid": False, "error": "VPD's causal-importance network is not on this machine", "steps": 0, "explanation": "", "notes": []})
                continue
            if src == "vpd":
                g = nat.vpd_answer(ids)
                rows.append((j, g.bits(positions, total), [g] * P))
                out.append({"valid": True, "error": None, "steps": 1, "explanation": "", "notes": [], "base": g})
                continue
            ir = mech.trace(src, "vpd4l", behavior=task)
            out.append({"valid": ir["valid"], "error": ir.get("error"), "steps": ir["graph"].get("steps", 0), "explanation": ir["graph"].get("explanation", ""),
                        "notes": ir["graph"].get("notes", []), "dropped": [why for _, why in ir["graph"].get("dropped", [])]})
            if not ir["valid"]:
                continue
            irs = self.instances(src, task, prompts)
            n = out[-1]["steps"]
            mine = []
            for k in range(1, n + 1):
                gs = []
                for x_ir in [ir] + irs:
                    if x_ir is None:
                        gs.append(self.native.Graph())
                        continue
                    g, uses = self.native.prefix(x_ir, min(k, x_ir["graph"]["steps"]))
                    if uses:
                        g = self.expand(g, uses, targets, len(ids))
                        if g is None:
                            out[-1].update(valid=False, error=f"uses: an entry of {uses} is unknown, outside the text or not a model connection")
                            break
                    gs.append(g)
                if not out[-1]["valid"]:
                    break
                mine.append((code_bits(prefix_code(src, k)), gs))
            if out[-1]["valid"]:
                step_nodes[j] = [gs[0].node_set() for _, gs in mine]
                for bits, gs in mine:
                    rows.append((j, bits, gs[1:]))
                out[-1]["base"] = mine[-1][1][0] if mine else self.native.Graph()
                if necessity and mine:
                    out[-1]["necessity_kl_bits"] = nat.necessity(ids, targets, out[-1]["base"], prompts)
        kl = nat.faithfulness(ids, targets, [[r[2][i] for r in rows] for i in range(P)], prompts, seed)
        holds = [[b - a for a, b in zip([set()] + st[:-1], st)] for st in step_nodes]  # each answer's steps: the nodes each adds
        events = nat.events(ids, targets, prompts, [[h for h in hs if h] for hs in holds], seed)  # the English reader's questions (reader.py)
        hi = positions * total * math.log2(positions * total)
        for s, ev in zip(out, events):
            s.update(curve=[[lo, kl[0]]], lo=lo, hi=max(hi, 2 * lo), events=ev)
        for (o, b, _), k in zip(rows[1:], kl[1:]):
            out[o]["curve"].append([b, k])
            out[o].update(kl_bits=k, bits=b)
        for s in out:
            g = s.pop("base", None)
            if g is not None:
                s.update(nodes=g.count(), edges=None if g.complete else g.edges())
            if s["valid"] and len(s["curve"]) == 1:  # no steps: the empty program
                s.update(kl_bits=kl[0], bits=lo, nodes=0, edges=0)
        return out

    def adversarial(self, tasks: list[dict], sources: list[str], seed: int = 0) -> list[float | None]:
        """native.adversarial over one answer per task (an answer's whole graph, library entries expanded; "vpd" for VPD's
        answer), one adversary shared by all of them: each task's KL in bits, None where the answer is invalid."""
        import mech

        cases, where = [], []
        for j, (task, src) in enumerate(zip(tasks, sources)):
            prompt = task["prompts"][0]
            ids, targets = prompt["token_ids"], prompt["target_positions"]
            if src == "vpd" and not self.nat.has_importance:
                continue
            if src == "vpd":
                g = self.nat.vpd_answer(ids)
            else:
                ir = mech.trace(src, "vpd4l", behavior=task)
                if not ir["valid"] or not ir["graph"]["steps"]:
                    continue
                g, uses = self.native.prefix(ir, ir["graph"]["steps"])
                g = self.expand(g, uses, targets, len(ids)) if uses else g
                if g is None:
                    continue
            cases.append((ids, targets, g))
            where.append(j)
        out = [None] * len(tasks)
        for j, k in zip(where, self.nat.adversarial(cases, seed=seed) if cases else []):
            out[j] = k
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
