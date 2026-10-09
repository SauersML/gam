"""One-edit neighbours of a format v3 explanation, scored by the checker (#2951 graph oracle): the edit step shared by
the teacher search's refinement and RL credit (rl/train.py).

An explanation's choices are its groups' subcomponents. An edit changes one choice: drop one subcomponent from a group
(a group left without subcomponents is removed), drop a whole group (its name leaves every other group's reads), or
add a candidate subcomponent to a group. The checker scores a batch of edited explanations on the current behavior
under the same experiments, and dS = S(edited) - S(explanation) is the edit's measured effect given the rest of the
explanation: a removal measured by running M, never a gradient. An invalid explanation scores +inf.

  credit(answer, score, k, rng)        -> {edit: dS} for k sampled subcomponent drops and every group drop
  refine(answer, score, candidates, R) -> the best explanation found, its score and the accepted edits

`score` takes a list of sources and returns their checker results (dicts with total_bits and valid), all under one
experiment draw: Checker.score_batch with fixed experiments and seed (an edit leaves the English
explanation as it is).

  edits.py credit ANSWER.py BEHAVIOR.json [--k 16] [--experiments 16] [--seed 0] [--vpd DIR]
  edits.py refine ANSWER.py BEHAVIOR.json --candidates CAND.json [--rounds 8] [--adds 8] [--out OUT.py]
CAND.json: {group: [subcomponent token, ...]} best first.
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
KIND = "group"  # the kind of every choice holder (rl/train.py's credit marks name it)


@dataclass(frozen=True)
class Statement:
    """One group: its line in the source (its key's line; -1 when an edit wrote it), name and fields."""

    line: int
    variable: str  # the group's name
    parts: tuple[str, ...]
    reads: tuple[str, ...]
    label: str | None
    writes: str | None
    kind: str = KIND


@dataclass(frozen=True)
class Answer:
    """An explanation: the lines before its `groups` statement, its groups, and the lines after."""

    head: tuple[str, ...]
    statements: tuple[Statement, ...]
    tail: tuple[str, ...]

    @staticmethod
    def parse(source: str) -> "Answer":
        """The groups of a source whose `groups` statement is a literal dict; none when it is not."""
        lines = tuple(source.rstrip("\n").split("\n"))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return Answer(lines, (), ())
        node = next((n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "groups" for t in n.targets)), None)
        if node is None or not isinstance(node.value, ast.Dict):
            return Answer(lines, (), ())
        try:
            groups = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return Answer(lines, (), ())
        statements = []
        for key, value in zip(node.value.keys, node.value.values):
            name = getattr(key, "value", None)
            g = groups.get(name)
            if not isinstance(name, str) or not isinstance(g, dict):
                return Answer(lines, (), ())
            parts = g.get("subcomponents")
            reads = g.get("reads") or []
            if not isinstance(parts, (list, tuple)) or not isinstance(reads, (list, tuple)):
                return Answer(lines, (), ())
            statements.append(Statement(key.lineno - 1, name, tuple(map(str, parts)), tuple(map(str, reads)),
                                        g.get("label"), g.get("writes")))
        return Answer(lines[: node.lineno - 1], tuple(statements), lines[node.end_lineno:])

    def source(self) -> str:
        """The explanation with its groups written one per line (explain.source's form)."""
        rows = ["groups = {"]
        for s in self.statements:
            fields = [f'"subcomponents": [{", ".join(json.dumps(p) for p in s.parts)}]',
                      f'"reads": [{", ".join(json.dumps(r) for r in s.reads)}]']
            if s.label:
                fields.append(f'"label": {json.dumps(s.label)}')
            if s.writes:
                fields.append(f'"writes": {json.dumps(s.writes)}')
            rows.append(f'    {json.dumps(s.variable)}: {{{", ".join(fields)}}},')
        rows.append("}")
        return "\n".join(list(self.head) + rows + list(self.tail)) + "\n"

    def parts(self) -> list[tuple[int, str]]:
        """Every (group index, subcomponent) choice."""
        return [(j, p) for j, s in enumerate(self.statements) for p in s.parts]


@dataclass(frozen=True)
class Edit:
    op: str  # drop / unalign (drop the whole group) / add
    variable: str  # the group's name
    kind: str = KIND
    part: str | None = None

    def __str__(self) -> str:
        return f"{self.op} {self.variable}{' ' + self.part if self.part else ''}"


def without_group(statements: list[Statement], name: str) -> list[Statement]:
    """`statements` less group `name`, its name gone from every other group's reads (a read of name:route too)."""
    return [replace(s, reads=tuple(r for r in s.reads if r.split(":")[0] != name)) for s in statements if s.variable != name]


def apply(answer: Answer, edit: Edit) -> Answer:
    """The explanation with one edit made."""
    statements = list(answer.statements)
    j = next((j for j, s in enumerate(statements) if s.variable == edit.variable), None)
    if j is None:
        return answer
    if edit.op == "add":
        if edit.part not in statements[j].parts:
            statements[j] = replace(statements[j], parts=statements[j].parts + (edit.part,))
    elif edit.op == "unalign":
        statements = without_group(statements, edit.variable)
    elif edit.op == "drop":
        kept = tuple(p for p in statements[j].parts if p != edit.part)
        statements = without_group(statements, edit.variable) if not kept else statements[:j] + [replace(statements[j], parts=kept)] + statements[j + 1:]
    return replace(answer, statements=tuple(statements))


def drops(answer: Answer) -> list[Edit]:
    return [Edit("drop", answer.statements[j].variable, part=p) for j, p in answer.parts()]


def unaligns(answer: Answer) -> list[Edit]:
    return [Edit("unalign", s.variable) for s in answer.statements]


def adds(answer: Answer, candidates: dict[str, list[str]], per_variable: int) -> list[Edit]:
    """The first `per_variable` candidates of each group that no group names yet."""
    named = {p for s in answer.statements for p in s.parts}
    groups = {s.variable for s in answer.statements}
    out = []
    for name, ranked in candidates.items():
        if name in groups:
            out += [Edit("add", name, part=p) for p in [p for p in ranked if p not in named][:per_variable]]
    return out


def totals(results: list[dict]) -> list[float]:
    return [r["total_bits"] if r.get("valid", True) else math.inf for r in results]


def credit_edits(answer: Answer, k: int = 16, rng: random.Random | None = None) -> list[Edit]:
    """credit's edits: k subcomponent drops sampled uniformly (all when fewer), then every group drop."""
    rng = rng or random.Random(0)
    part_drops = drops(answer)
    if len(part_drops) > k:
        part_drops = rng.sample(part_drops, k)
    return part_drops + unaligns(answer)


def credit(answer: Answer, score, k: int = 16, rng: random.Random | None = None) -> tuple[float, dict[Edit, float]]:
    """The explanation's score and dS for k subcomponent drops sampled uniformly and every group drop, one batch."""
    edits = credit_edits(answer, k, rng)
    s = totals(score([answer.source()] + [apply(answer, e).source() for e in edits]))
    return s[0], {e: v - s[0] for e, v in zip(edits, s[1:])}


def refine(answer: Answer, score, candidates: dict[str, list[str]] | None = None, rounds: int = 8,
           per_variable: int = 8, log=None, max_drops: int | None = None,
           rng: random.Random | None = None) -> tuple[Answer, float, list[Edit]]:
    """Greedy improvement: each round scores every subcomponent drop (`max_drops` of them sampled uniformly when the
    explanation has more) and the next `per_variable` candidates of each group in one batch, then makes the improving
    edits jointly in order of dS, trying all of them, half, a quarter, ... and the best single one in a second batch,
    and keeps the lowest score. Stops when no edit improves or after `rounds` rounds."""
    rng = rng or random.Random(0)
    current = totals(score([answer.source()]))[0]
    accepted: list[Edit] = []
    for r in range(rounds):
        part_drops = drops(answer)
        if max_drops is not None and len(part_drops) > max_drops:
            part_drops = rng.sample(part_drops, max_drops)
        edits = part_drops + adds(answer, candidates or {}, per_variable)
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


def checker_score(behavior_path: Path, vpd: Path, experiments: int, seed: int, device: str | None = "gpu"):
    """A Checker on the behavior and its score function (the caller closes the checker); device "gpu" runs on the
    single-precision device (None: the host, float64)."""
    import score as score_module

    behavior = json.loads(behavior_path.read_text())
    c = score_module.Checker(behavior["model"], views={"vpd": vpd}, device=device)
    c.behavior(behavior_path)
    return c, lambda sources: c.score_batch(sources, experiments=experiments, seed=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["credit", "refine"])
    ap.add_argument("answer", type=Path, help="the explanation's source (.py)")
    ap.add_argument("behavior", type=Path)
    ap.add_argument("--candidates", type=Path, help="refine: {group: [subcomponent token, ...]} best first")
    ap.add_argument("--k", type=int, default=16, help="credit: subcomponent drops sampled")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--adds", type=int, default=8, help="refine: candidates tried per group per round")
    ap.add_argument("--experiments", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vpd", type=Path, default=Path.home() / "mpd-data/engine/vpd4l_decomposition")
    ap.add_argument("--device", default="gpu", help="gpu (the single-precision device) or host")
    ap.add_argument("--out", type=Path, help="refine: where the refined explanation goes")
    a = ap.parse_args()
    answer = Answer.parse(a.answer.read_text())
    checker, score = checker_score(a.behavior, a.vpd, a.experiments, a.seed, None if a.device == "host" else a.device)
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
