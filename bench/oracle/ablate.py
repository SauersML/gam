"""Wrong-relationship ablations of an oracle report's rule (#2951): the same concepts with a wrong
relation, to show that a reader's score depends on the relation the rule states, not on its words.

For each kind of change, a rewriter LLM gets an explicit instruction:
  reversed  the direction of one relation is reversed (increases <-> decreases, promotes <-> suppresses,
            X leads to Y <-> Y leads to X), every concept kept;
  swapped   the roles of two variables are exchanged (what the rule says of X it says of Y, and back);
  permuted  a mapping is permuted (pairs of items, such as conditions and responses, reassigned so that
            every item still appears and no original pair survives).
It answers that the kind does not apply when the rule has no such relation. A rewrite is kept only when
it passes three checks:
  entities  the same set of named entities: quoted spans, numbers, and capitalized words that do not
            start a sentence, and identifiers with digits or underscores;
  length    word counts within a factor of --length-factor of each other (default 1.25: a relation
            change replaces a few words of a rule of a sentence or two);
  relation  a second, independent call (a judge that sees both texts and not the instruction) says they
            do not state the same relations, and classifies the difference as the requested kind.
The rewriter and judge are headless Claude Code (reader.claude_json, no tools, structured JSON).

  ablate.py --report REPORT.json --out ABLATIONS.jsonl --model MODEL   (a report, or an episode file)
Output lines: {report_sha256, kind, original, rewritten, changed_relation, applicable, checks, passed,
rewriter, judge}. episodes.py score --documents ablated:KIND reads the passed ones.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from episodes import SCHEMA, canonical, sha256_hex
from reader import claude_json

KINDS = {
    "reversed": "Reverse the direction of exactly one relation the text states: an increase becomes a decrease, "
    "promoting becomes suppressing, raising becomes lowering, or 'X leads to Y' becomes 'Y leads to X'.",
    "swapped": "Exchange the roles of exactly two variables or concepts the text names: everything the text says "
    "of the first it now says of the second, and the reverse.",
    "permuted": "Permute exactly one mapping the text states (pairs such as conditions and responses, inputs and "
    "outputs, or components and functions): reassign the pairs so that every item still appears and no "
    "original pair is kept.",
}

SYSTEM = "You rewrite and compare short technical texts exactly as instructed."

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "applicable": {"type": "boolean"},
        "rewritten": {"type": "string"},
        "changed_relation": {"type": "string"},
    },
    "required": ["applicable", "rewritten", "changed_relation"],
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "same_relations": {"type": "boolean"},
        "difference": {"enum": ["reversed", "swapped", "permuted", "other", "none"]},
        "explanation": {"type": "string"},
    },
    "required": ["same_relations", "difference", "explanation"],
    "additionalProperties": False,
}


def rewrite_prompt(rule: str, kind: str) -> str:
    return (
        f"Text:\n<<<{rule}>>>\n\n"
        f"Rewrite the text so that it names exactly the same concepts, entities, tokens, numbers, layers, heads "
        f"and quantities, but states a wrong relation among them, made by this change: {KINDS[kind]} Change "
        f"nothing else; keep the wording and length as close to the original as the change allows. If the text "
        f"states no relation of this kind, set applicable to false and rewritten to the empty string. In "
        f"changed_relation, say in one sentence which relation you changed and how."
    )


def judge_prompt(a: str, b: str) -> str:
    return (
        f"Text A:\n<<<{a}>>>\n\nText B:\n<<<{b}>>>\n\n"
        "Do the two texts state the same relations among the same concepts? If they do not, classify the "
        "difference: 'reversed' (the direction of a relation is reversed), 'swapped' (the roles of two "
        "variables are exchanged), 'permuted' (a mapping between items is reassigned), or 'other'. If they do, "
        "the difference is 'none'. Explain in one sentence."
    )


def entities(text: str) -> set[str]:
    found = []
    found += re.findall(r"\"[^\"]+\"|'[^']+'|`[^`]+`|“[^”]+”|‘[^’]+’", text)
    found += re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text)
    found += re.findall(r"\b\w*(?:\d\w*_|_\w*\d|[A-Za-z]\d|\d[A-Za-z])\w*\b", text)
    for sentence in re.split(r"(?<=[.!?:;])\s+|\n+", text):
        words = re.findall(r"[A-Za-z][\w'-]*", sentence)
        found += [w for w in words[1:] if w[0].isupper()]
    return set(found)


def words(text: str) -> int:
    return len(text.split())


def ablate(rule: str, report_sha256: str, model: str, length_factor: float) -> list[dict]:
    out = []
    for kind in KINDS:
        rewrite, rewriter = claude_json(rewrite_prompt(rule, kind), REWRITE_SCHEMA, model, SYSTEM)
        row = {"report_sha256": report_sha256, "kind": kind, "original": rule, "rewritten": rewrite["rewritten"],
               "changed_relation": rewrite["changed_relation"], "applicable": rewrite["applicable"], "rewriter": rewriter}
        if not rewrite["applicable"] or not rewrite["rewritten"].strip():
            row.update(checks={}, passed=False, judge=None)
            out.append(row)
            continue
        new = rewrite["rewritten"]
        verdict, judge = claude_json(judge_prompt(rule, new), JUDGE_SCHEMA, model, SYSTEM)
        before, after = entities(rule), entities(new)
        ratio = max(words(rule), words(new)) / max(1, min(words(rule), words(new)))
        checks = {
            "entities": before == after,
            "entities_only_original": sorted(before - after),
            "entities_only_rewritten": sorted(after - before),
            "length_ratio": ratio,
            "length": ratio <= length_factor,
            "relation": (not verdict["same_relations"]) and verdict["difference"] == kind and new.strip() != rule.strip(),
            "judge_difference": verdict["difference"],
        }
        row.update(checks=checks, passed=checks["entities"] and checks["length"] and checks["relation"], judge={**verdict, **judge})
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True, help="a report JSON, or an episode file (its frozen report)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="the rewriter and judge model for claude -p (e.g. sonnet)")
    ap.add_argument("--length-factor", type=float, default=1.25)
    args = ap.parse_args()
    obj = json.loads(Path(args.report).read_text())
    if obj.get("schema") == SCHEMA:
        report, sha = obj["report"]["content"], obj["report"]["sha256"]
    else:
        report, sha = obj, sha256_hex(canonical(obj))
    rows = ablate(report["rule"], sha, args.model, args.length_factor)
    with open(args.out, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps([{"kind": r["kind"], "passed": r["passed"], "rewritten": r["rewritten"], "checks": r["checks"]} for r in rows], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
