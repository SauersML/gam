"""Teacher programs (#2951 graph oracle): a search's program (nodes of decomposition parts, e2e/search.py) and
its behavior's family algorithm (algorithms/index.json) -> algorithm programs whose variables are aligned to
the search's nodes. e2e/teacher_run.py scores every assignment, refines the best (edits.refine) and prints it
(printer.py) in the oracle's answer format. Held-out families have no algorithm here: algorithm_of refuses a
behavior outside the train split, so no held-out behavior becomes a training answer.

Assignments: the algorithm's aligned variables, in data-flow order, take the program's nodes in layer order,
each a contiguous run of nodes that writes the residual stream (the answer takes the last run). A variable
whose values are attention patterns (lists of positions) is claimed on the query and key parts of the
attention nodes of the aligned variable that reads it.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402

SEARCH_HEADER = re.compile(r"PD\.vpd\[(\d+)\]")  # searches written before mech's generic PD[l] spelling
ANY = {"vpd4l": "<p:0.v.0>, <p:0.o.0>", "qwen3-0.6b": "<p:0.h.0>"}  # parts that write the residual


def algorithm_of(behavior: dict) -> str:
    """The family algorithm's source for a train behavior."""
    if behavior.get("split") != "train":
        raise ValueError(f"{behavior.get('id')}: split {behavior.get('split')}; held-out behaviors never become training answers")
    index = json.loads((HERE / "algorithms/index.json").read_text())
    if behavior["family"] not in index:
        raise KeyError(f"no algorithm for family {behavior['family']}")
    return (HERE / "algorithms" / f"{index[behavior['family']]}.py").read_text()


def search_ir(search: dict, model: str) -> dict:
    """The traced program of a search result (its source, in mech's current spelling)."""
    ir = mech.trace_inline(SEARCH_HEADER.sub(r"PD[\1]", search["source"]), model)
    if not ir["valid"]:
        raise ValueError(f"the search's program does not trace: {ir['error']}")
    return ir


def patterns(algorithm: str, behavior: dict) -> set[str]:
    """The algorithm's variables whose values are attention patterns: at every position a list of
    positions up to it, on the behavior's first prompt."""
    ir = mech.trace_inline(algorithm + f"\nalign(answer, {ANY[behavior['model']]})\n", behavior["model"])
    names = [v["name"] for v in ir["variables"]]
    payload, _ = mech.behavior_tokens(behavior, behavior["model"])
    namespace = {"__builtins__": mech.SAFE_BUILTINS}
    tree = mech.ast.fix_missing_locations(mech._NoImports().visit(mech.ast.parse(mech.quote_parts(algorithm))))
    exec(compile(tree, "<algorithm>", "exec"), namespace)
    values = mech._Algorithm(namespace, names).values(payload["prompts"][0], names)
    return {n for n, vs in values.items()
            if all(isinstance(v, list) and all(isinstance(j, int) and 0 <= j <= t for j in v) for t, v in enumerate(vs))}


def site(node: dict) -> tuple[int, int]:
    p = node["pieces"][0]
    return p["layer"], p["kind"] in ("mlp", "c_fc", "down_proj", "feature")


def writes(node: dict) -> bool:
    return any(p["kind"] in ("o_proj", "down_proj", "head", "attn", "mlp", "feature") for p in node["pieces"])


def tokens(pieces: list[dict]) -> str:
    import printer

    return ", ".join(printer.address(p) for p in pieces)


def assignments(ir: dict, algorithm: str, behavior: dict) -> list[str]:
    """Every algorithm program aligning the search's nodes to the algorithm's variables (module doc): any
    subset of its value variables that includes the answer is aligned (the rest stay unaligned steps); the
    programs mech rejects (a variable's parts must write before its readers read) are left out."""
    model = behavior["model"]
    traced = mech.trace_inline(algorithm + f"\nalign(answer, {ANY[model]})\n", model)
    claimed = patterns(algorithm, behavior)
    values = [v["name"] for v in traced["variables"] if v["name"] not in claimed and v["name"] != "answer"]
    reads = {v["name"]: set(v["reads"]) for v in traced["variables"]}
    nodes = sorted(ir["nodes"], key=site)
    out = []
    for r in range(len(values) + 1):
        for chosen in itertools.combinations(values, r):
            aligned = list(chosen) + ["answer"]
            for cuts in itertools.combinations(range(1, len(nodes)), len(aligned) - 1):
                runs = [nodes[a:b] for a, b in zip((0,) + cuts, cuts + (len(nodes),))]
                if not all(any(writes(n) for n in run) for run in runs):
                    continue
                lines = []
                for name, run in zip(aligned, runs):
                    lines.append(f"align({name}, {tokens([p for n in run for p in n['pieces']])})")
                    for c in sorted(claimed & reads[name]):
                        qk = [p for n in run for p in n["pieces"] if p["kind"] in ("q_proj", "k_proj", "head")]
                        if qk:
                            lines.append(f"claim({c}, {tokens(qk)})")
                source = algorithm.rstrip() + "\n\n\n" + "\n".join(lines) + "\n"
                if mech.trace_inline(source, model, decomposition=ir.get("decomposition") or "native")["valid"]:
                    out.append(source)
    return out
