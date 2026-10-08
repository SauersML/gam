"""One-edit neighbours of an oracle answer, scored by the checker (#2951 graph oracle): the edit step shared by
teacher refinement (teacher.py), the search's refinement (e2e/vpd_min.py) and RL credit (rl/train.py).

An answer's choices are its statements align(variable, parts...) and claim(variable, parts...). An edit changes one
choice: drop one part from a statement (a statement left without parts is removed), drop a variable's statement, or
add a candidate part to a variable's statement (a new align statement when the variable has none). The checker
scores a batch of edited answers on the current behavior under the same experiments, and
dS = S(edited) - S(answer) is the edit's measured effect given the rest of the answer: a removal measured by
running M, never a gradient. An invalid answer scores +inf.

  credit(answer, score, k, rng)        -> {edit: dS} for k sampled part drops and every statement drop
  refine(answer, score, candidates, R) -> the best answer found, its score and the accepted edits

`score` takes a list of answer sources and returns their checker results (dicts with total_bits and valid), all
under one experiment draw: Checker.score_batch with fixed experiments and seed and the reader off (an edit leaves
the English explanation as it is).

  edits.py credit ANSWER.py BEHAVIOR.json [--k 16] [--experiments 16] [--seed 0] [--vpd DIR]
  edits.py refine ANSWER.py BEHAVIOR.json --candidates CAND.json [--rounds 8] [--adds 8] [--out OUT.py]
CAND.json: {variable: [part token, ...]} best first (for example VPD importance on the variable's interchange pairs).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PART = re.compile(r"<p:[^>]+>")
STATEMENT = re.compile(r"^(\s*)(align|bind|claim)\(\s*(\w+)\s*,(.*)\)\s*$")


def align_keyword() -> str:
    """The alignment statement's name in mech's current answer format."""
    import mech

    return "align" if hasattr(mech, "align") else "bind"


@dataclass(frozen=True)
class Statement:
    line: int  # its line in the source (-1: added by an edit, written after the last statement)
    kind: str  # align (or bind) / claim
    variable: str
    parts: tuple[str, ...]


@dataclass(frozen=True)
class Answer:
    """An answer's source, split into the lines no edit touches and its statements."""

    lines: tuple[str, ...]
    statements: tuple[Statement, ...]
    keyword: str = field(default="align")

    @staticmethod
    def parse(source: str, keyword: str | None = None) -> "Answer":
        lines = tuple(source.rstrip("\n").split("\n"))
        statements = []
        for i, text in enumerate(lines):
            m = STATEMENT.match(text)
            if m and PART.search(m[4]):
                statements.append(Statement(i, m[2], m[3], tuple(PART.findall(m[4]))))
        found = next((s.kind for s in statements if s.kind != "claim"), None)
        return Answer(lines, tuple(statements), keyword or found or align_keyword())

    def source(self) -> str:
        by_line = {s.line: s for s in self.statements if s.line >= 0}
        out = []
        for i, text in enumerate(self.lines):
            if i in by_line:
                s = by_line[i]
                out.append(f"{s.kind}({s.variable}, {', '.join(s.parts)})")
            elif not STATEMENT.match(text) or not PART.search(text):
                out.append(text)
        out += [f"{s.kind}({s.variable}, {', '.join(s.parts)})" for s in self.statements if s.line < 0]
        return "\n".join(out) + "\n"

    def parts(self) -> list[tuple[int, str]]:
        """Every (statement index, part) choice."""
        return [(j, p) for j, s in enumerate(self.statements) for p in s.parts]


@dataclass(frozen=True)
class Edit:
    op: str  # drop / unalign / add
    variable: str
    kind: str
    part: str | None = None

    def __str__(self) -> str:
        return f"{self.op} {self.kind}({self.variable}{', ' + self.part if self.part else ''})"


def apply(answer: Answer, edit: Edit) -> Answer:
    """The answer with one edit made."""
    statements = list(answer.statements)
    j = next((j for j, s in enumerate(statements) if s.variable == edit.variable and s.kind == edit.kind), None)
    if edit.op == "add":
        if j is None:
            statements.append(Statement(-1, edit.kind, edit.variable, (edit.part,)))
        elif edit.part not in statements[j].parts:
            statements[j] = replace(statements[j], parts=statements[j].parts + (edit.part,))
    elif j is not None and edit.op == "unalign":
        del statements[j]
    elif j is not None and edit.op == "drop":
        kept = tuple(p for p in statements[j].parts if p != edit.part)
        if kept:
            statements[j] = replace(statements[j], parts=kept)
        else:
            del statements[j]
    return replace(answer, statements=tuple(statements))


def drops(answer: Answer) -> list[Edit]:
    return [Edit("drop", answer.statements[j].variable, answer.statements[j].kind, p) for j, p in answer.parts()]


def unaligns(answer: Answer) -> list[Edit]:
    return [Edit("unalign", s.variable, s.kind) for s in answer.statements]


def adds(answer: Answer, candidates: dict[str, list[str]], per_variable: int) -> list[Edit]:
    """The first `per_variable` candidates of each variable that its statement does not name yet."""
    named = {(s.variable, s.kind): set(s.parts) for s in answer.statements}
    kinds = {s.variable: s.kind for s in answer.statements}
    out = []
    for variable, ranked in candidates.items():
        kind = kinds.get(variable, answer.keyword)
        fresh = [p for p in ranked if p not in named.get((variable, kind), ())]
        out += [Edit("add", variable, kind, p) for p in fresh[:per_variable]]
    return out


def totals(results: list[dict]) -> list[float]:
    return [r["total_bits"] if r.get("valid", True) else math.inf for r in results]


def credit(answer: Answer, score, k: int = 16, rng: random.Random | None = None) -> tuple[float, dict[Edit, float]]:
    """The answer's score and dS for k part drops sampled uniformly (all when fewer) and every statement drop,
    one checker batch."""
    rng = rng or random.Random(0)
    part_drops = drops(answer)
    if len(part_drops) > k:
        part_drops = rng.sample(part_drops, k)
    edits = part_drops + unaligns(answer)
    s = totals(score([answer.source()] + [apply(answer, e).source() for e in edits]))
    return s[0], {e: v - s[0] for e, v in zip(edits, s[1:])}


def refine(answer: Answer, score, candidates: dict[str, list[str]] | None = None, rounds: int = 8,
           per_variable: int = 8, log=None) -> tuple[Answer, float, list[Edit]]:
    """Greedy improvement: each round scores every part drop and the next `per_variable` candidates of each
    variable in one batch, then makes the improving edits jointly in order of dS, trying all of them, half, a
    quarter, ... and the best single one in a second batch, and keeps the lowest score. Stops when no edit improves
    or after `rounds` rounds."""
    current = totals(score([answer.source()]))[0]
    accepted: list[Edit] = []
    for r in range(rounds):
        edits = drops(answer) + adds(answer, candidates or {}, per_variable)
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


def checker_score(behavior_path: Path, vpd: Path, experiments: int, seed: int):
    """A Checker on the behavior and its score function (the caller closes the checker)."""
    import score as score_module

    behavior = json.loads(behavior_path.read_text())
    views = {"vpd": vpd} if behavior["model"] == "vpd4l" else None
    c = score_module.Checker(behavior["model"], views=views)
    c.behavior(behavior_path)
    return c, lambda sources: c.score_batch(sources, experiments=experiments, seed=seed, reader=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["credit", "refine"])
    ap.add_argument("answer", type=Path, help="the answer's program source (.py)")
    ap.add_argument("behavior", type=Path)
    ap.add_argument("--candidates", type=Path, help="refine: {variable: [part token, ...]} best first")
    ap.add_argument("--k", type=int, default=16, help="credit: part drops sampled")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--adds", type=int, default=8, help="refine: candidates tried per variable per round")
    ap.add_argument("--experiments", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vpd", type=Path, default=Path.home() / "mpd-data/engine/vpd4l_decomposition")
    ap.add_argument("--out", type=Path, help="refine: where the refined program goes")
    a = ap.parse_args()
    answer = Answer.parse(a.answer.read_text())
    checker, score = checker_score(a.behavior, a.vpd, a.experiments, a.seed)
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
