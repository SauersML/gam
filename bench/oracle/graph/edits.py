"""One-edit neighbours of a format v4 explanation, scored by the checker (#2951 graph oracle): the edit step shared by
the teacher search's refinement and RL credit (rl/train.py).

An explanation's choices are its nodes' subcomponents and its edges. An edit changes one choice: drop one subcomponent
from a node (a node left without subcomponents is removed with its edges and label), drop a whole node, cut one edge,
or add a candidate subcomponent to a node. The checker scores a batch of edited explanations on the current behavior
under the same experiments, and dS = S(edited) - S(explanation) is the edit's measured effect given the rest of the
explanation: a removal measured by running M, never a gradient. An invalid explanation scores +inf.

  credit(answer, score, k, rng)        -> {edit: dS} for k sampled subcomponent drops, every node drop and edge cut
  refine(answer, score, candidates, R) -> the best explanation found, its score and the accepted edits

`score` takes a list of sources and returns their checker results (dicts with total_bits and valid), all under one
experiment draw: Checker.score_batch with fixed experiments and seed (an edit leaves the English as it is).

The canonical form (Answer.source; the teacher answers and the oracle's grammar): the code before `nodes` as written,
then one node per line, one edge per line, and the labels on one line:

    nodes = {
        "mark": {"subcomponents": ["<p:0.fc.225>", "<p:0.down.663>"], "at": quote_marks},
    }
    edges = [
        ("input", "mark"),
        ("mark", "output"),
    ]
    labels = {"mark": "inside"}

  edits.py credit ANSWER.py BEHAVIOR.json [--k 16] [--experiments 16] [--seed 0]
  edits.py refine ANSWER.py BEHAVIOR.json --candidates CAND.json [--rounds 8] [--adds 8] [--out OUT.py]
CAND.json: {node: [subcomponent token, ...]} best first.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402

PART = mech.PART
KIND = "node"  # the kind of every choice holder (rl/train.py's credit marks name it)


@dataclass(frozen=True)
class Statement:
    """One node: its line in the source (its key's line; -1 when an edit wrote it), name, subcomponents and where it
    acts ("all", "targets", "last", or the name of a function of the file)."""

    line: int
    variable: str  # the node's name
    parts: tuple[str, ...]
    at: str = "all"
    kind: str = KIND


@dataclass(frozen=True)
class Answer:
    """An explanation: the lines before its `nodes` statement, its nodes, edges and labels, and the lines after."""

    head: tuple[str, ...]
    statements: tuple[Statement, ...]
    edges: tuple[tuple[str, ...], ...] = ()
    labels: tuple[tuple[str, str], ...] = ()
    tail: tuple[str, ...] = ()
    edge_lines: tuple[int, ...] = ()  # each edge's line in the parsed source (-1: written by an edit)

    @staticmethod
    def parse(source: str) -> "Answer":
        """The nodes, edges and labels of a source whose `nodes` statement is a dict of literal entries (an "at" may name
        a function) and whose `edges` and `labels` are literals; no nodes when they are not."""
        lines = tuple(source.rstrip("\n").split("\n"))
        none = Answer(lines, ())
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return none
        stmts = {t.id: n for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name) and t.id in mech.STRUCTURE}
        if "nodes" not in stmts or "edges" not in stmts or not isinstance(stmts["nodes"].value, ast.Dict):
            return none
        statements = []
        for key, value in zip(stmts["nodes"].value.keys, stmts["nodes"].value.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str) or not isinstance(value, ast.Dict):
                return none
            fields = {k.value: v for k, v in zip(value.keys, value.values) if isinstance(k, ast.Constant)}
            try:
                parts = ast.literal_eval(fields["subcomponents"])
            except (KeyError, ValueError):
                return none
            at = fields.get("at")
            where = "all" if at is None else at.value if isinstance(at, ast.Constant) and isinstance(at.value, str) else at.id if isinstance(at, ast.Name) else None
            if where is None or not isinstance(parts, (list, tuple)):
                return none
            statements.append(Statement(key.lineno - 1, key.value, tuple(map(str, parts)), where))
        try:
            edges = ast.literal_eval(stmts["edges"].value)
            labels = ast.literal_eval(stmts["labels"].value) if "labels" in stmts else {}
        except ValueError:
            return none
        if not isinstance(edges, (list, tuple)) or not isinstance(labels, dict):
            return none
        edge_lines = tuple(e.lineno - 1 for e in stmts["edges"].value.elts) if isinstance(stmts["edges"].value, (ast.List, ast.Tuple)) else ()
        first = min(n.lineno for n in stmts.values())
        last = max(n.end_lineno for n in stmts.values())
        return Answer(lines[: first - 1], tuple(statements), tuple(tuple(map(str, e)) for e in edges),
                      tuple((str(k), str(v)) for k, v in labels.items()), lines[last:], edge_lines)

    def source(self) -> str:
        """The explanation in the canonical form (module doc)."""
        quote = lambda w: w if w not in ("all", "targets", "last") else json.dumps(w)  # noqa: E731
        rows = ["nodes = {"]
        rows += [f'    {json.dumps(s.variable)}: {{"subcomponents": [{", ".join(json.dumps(p) for p in s.parts)}], "at": {quote(s.at)}}},'
                 for s in self.statements]
        rows += ["}", "edges = ["] + [f"    ({', '.join(json.dumps(x) for x in e)})," for e in self.edges] + ["]"]
        if self.labels:
            rows.append("labels = {" + ", ".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in self.labels) + "}")
        head = list(self.head)
        while head and not head[-1].strip():
            head.pop()
        return "\n".join(head + (["", ""] if head else []) + rows + list(self.tail)) + "\n"

    def parts(self) -> list[tuple[int, str]]:
        """Every (node index, subcomponent) choice."""
        return [(j, p) for j, s in enumerate(self.statements) for p in s.parts]


@dataclass(frozen=True)
class Edit:
    op: str  # drop (a subcomponent) / unalign (drop the whole node) / cut (an edge) / add (a subcomponent)
    variable: str  # the node's name; for a cut, the edge as "writer>reader[>route]"
    kind: str = KIND
    part: str | None = None

    def __str__(self) -> str:
        return f"{self.op} {self.variable}{' ' + self.part if self.part else ''}"


def edge_name(e: tuple[str, ...]) -> str:
    return ">".join(e)


def without_node(answer: Answer, name: str) -> Answer:
    """`answer` less node `name`, its edges and its label."""
    return replace(answer, statements=tuple(s for s in answer.statements if s.variable != name),
                   edges=tuple(e for e in answer.edges if name not in e[:2]), labels=tuple((k, v) for k, v in answer.labels if k != name),
                   edge_lines=())


def apply(answer: Answer, edit: Edit) -> Answer:
    """The explanation with one edit made."""
    if edit.op == "cut":
        return replace(answer, edges=tuple(e for e in answer.edges if edge_name(e) != edit.variable), edge_lines=())
    statements = list(answer.statements)
    j = next((j for j, s in enumerate(statements) if s.variable == edit.variable), None)
    if j is None:
        return answer
    if edit.op == "add":
        if edit.part not in statements[j].parts:
            statements[j] = replace(statements[j], parts=statements[j].parts + (edit.part,))
        return replace(answer, statements=tuple(statements))
    if edit.op == "unalign":
        return without_node(answer, edit.variable)
    if edit.op == "drop":
        kept = tuple(p for p in statements[j].parts if p != edit.part)
        if not kept:
            return without_node(answer, edit.variable)
        statements[j] = replace(statements[j], parts=kept)
        return replace(answer, statements=tuple(statements))
    return answer


def drops(answer: Answer) -> list[Edit]:
    return [Edit("drop", answer.statements[j].variable, part=p) for j, p in answer.parts()]


def unaligns(answer: Answer) -> list[Edit]:
    return [Edit("unalign", s.variable) for s in answer.statements]


def cuts(answer: Answer) -> list[Edit]:
    return [Edit("cut", edge_name(e)) for e in answer.edges]


def adds(answer: Answer, candidates: dict[str, list[str]], per_variable: int) -> list[Edit]:
    """The first `per_variable` candidates of each node that no node names yet."""
    named = {p for s in answer.statements for p in s.parts}
    nodes = {s.variable for s in answer.statements}
    out = []
    for name, ranked in candidates.items():
        if name in nodes:
            out += [Edit("add", name, part=p) for p in [p for p in ranked if p not in named][:per_variable]]
    return out


def totals(results: list[dict]) -> list[float]:
    return [r["total_bits"] if r.get("valid", True) else math.inf for r in results]


def credit_edits(answer: Answer, k: int = 16, rng: random.Random | None = None) -> list[Edit]:
    """credit's edits: k subcomponent drops sampled uniformly (all when fewer), then every node drop and edge cut."""
    rng = rng or random.Random(0)
    part_drops = drops(answer)
    if len(part_drops) > k:
        part_drops = rng.sample(part_drops, k)
    return part_drops + unaligns(answer) + cuts(answer)


def credit(answer: Answer, score, k: int = 16, rng: random.Random | None = None) -> tuple[float, dict[Edit, float]]:
    """The explanation's score and dS for k subcomponent drops, every node drop and every edge cut, one batch."""
    edits = credit_edits(answer, k, rng)
    s = totals(score([answer.source()] + [apply(answer, e).source() for e in edits]))
    return s[0], {e: v - s[0] for e, v in zip(edits, s[1:])}


def refine(answer: Answer, score, candidates: dict[str, list[str]] | None = None, rounds: int = 8,
           per_variable: int = 8, log=None, max_drops: int | None = None,
           rng: random.Random | None = None) -> tuple[Answer, float, list[Edit]]:
    """Greedy improvement: each round scores every subcomponent drop (`max_drops` of them sampled uniformly when the
    explanation has more), every edge cut and the next `per_variable` candidates of each node in one batch, then makes
    the improving edits jointly in order of dS, trying all of them, half, a quarter, ... and the best single one in a
    second batch, and keeps the lowest score. Stops when no edit improves or after `rounds` rounds."""
    rng = rng or random.Random(0)
    current = totals(score([answer.source()]))[0]
    accepted: list[Edit] = []
    for r in range(rounds):
        part_drops = drops(answer)
        if max_drops is not None and len(part_drops) > max_drops:
            part_drops = rng.sample(part_drops, max_drops)
        edits = part_drops + cuts(answer) + adds(answer, candidates or {}, per_variable)
        if not edits:
            break
        s = totals(score([apply(answer, e).source() for e in edits]))
        better = sorted((v - current, i) for i, v in enumerate(s) if v < current)
        if not better:
            break
        sizes = sorted({len(better) >> h for h in range(len(better).bit_length())} | {1}, reverse=True)
        tries = []
        for n in sizes:
            joint = answer
            for _, i in better[:n]:
                joint = apply(joint, edits[i])
            tries.append((n, joint))
        joint_scores = (totals(score([t.source() for _, t in tries[:-1]])) if len(tries) > 1 else []) + [s[better[0][1]]]
        best = min(range(len(tries)), key=lambda t: joint_scores[t])
        answer, current = tries[best][1], joint_scores[best]
        accepted += [edits[i] for _, i in better[:tries[best][0]]]
        if log:
            log(f"round {r}: {len(edits)} edits scored, {len(better)} improve, kept {tries[best][0]} -> {current:.6g} bits")
    return answer, current, accepted


def checker_score(behavior_path: Path, experiments: int, seed: int, device: str | None = "gpu"):
    """A Checker on the behavior and its score function (the caller closes the checker); device "gpu" runs on the
    single-precision device (None: the host, float64)."""
    import score as score_module

    behavior = json.loads(behavior_path.read_text())
    c = score_module.Checker(behavior["model"], device=device)
    c.behavior(behavior_path)
    return c, lambda sources: c.score_batch(sources, experiments=experiments, seed=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["credit", "refine"])
    ap.add_argument("answer", type=Path, help="the explanation's source (.py)")
    ap.add_argument("behavior", type=Path)
    ap.add_argument("--candidates", type=Path, help="refine: {node: [subcomponent token, ...]} best first")
    ap.add_argument("--k", type=int, default=16, help="credit: subcomponent drops sampled")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--adds", type=int, default=8, help="refine: candidates tried per node per round")
    ap.add_argument("--experiments", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="gpu", help="gpu (the single-precision device) or host")
    ap.add_argument("--out", type=Path, help="refine: where the refined explanation goes")
    a = ap.parse_args()
    answer = Answer.parse(a.answer.read_text())
    checker, score = checker_score(a.behavior, a.experiments, a.seed, None if a.device == "host" else a.device)
    try:
        if a.command == "credit":
            s, dS = credit(answer, score, a.k)
            print(json.dumps({"total_bits": s, "dS": {str(e): v for e, v in sorted(dS.items(), key=lambda kv: kv[1])}}, indent=1))
        else:
            candidates = json.loads(a.candidates.read_text()) if a.candidates else {}
            best, s, accepted = refine(answer, score, candidates, a.rounds, a.adds, log=lambda m: print(m, file=sys.stderr))
            (a.out or a.answer.with_suffix(".refined.py")).write_text(best.source())
            print(json.dumps({"total_bits": s, "accepted": [str(e) for e in accepted]}, indent=1))
    finally:
        checker.close()


if __name__ == "__main__":
    main()
