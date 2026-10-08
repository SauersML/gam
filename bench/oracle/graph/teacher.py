"""Teacher programs (#2951 graph oracle): a search's program (nodes of decomposition parts, e2e/search.py) and
its behavior's family algorithm (algorithms/index.json) -> algorithm programs whose variables are bound to
the search's nodes; the checker scores every assignment and the best is printed with its facts and English
(printer.py), in the oracle's answer format.

Assignments: the algorithm's bound variables, in data-flow order, take the program's nodes in layer order,
each a contiguous run of nodes that writes the residual stream (the answer takes the last run). A variable
whose values are attention patterns (lists of positions) is claimed on the query and key parts of the
attention nodes of the bound variable that reads it.

  teacher.py SEARCH.json BEHAVIOR.json [--out-dir DIR] [--vpd DIR]
writes DIR/<behavior>.py, .answer.txt, .graph.json and .assignments.jsonl (every assignment's score).
"""

from __future__ import annotations

import argparse
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
    """The family algorithm's source for a behavior."""
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
    ir = mech.trace_inline(algorithm + f"\nbind(answer, {ANY[behavior['model']]})\n", behavior["model"])
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
    """Every algorithm program binding the search's nodes to the algorithm's variables (module doc): any
    subset of its value variables that includes the answer is bound (the rest stay unbound steps); the
    programs mech rejects (a variable's parts must write before its readers read) are left out."""
    model = behavior["model"]
    traced = mech.trace_inline(algorithm + f"\nbind(answer, {ANY[model]})\n", model)
    claimed = patterns(algorithm, behavior)
    values = [v["name"] for v in traced["variables"] if v["name"] not in claimed and v["name"] != "answer"]
    reads = {v["name"]: set(v["reads"]) for v in traced["variables"]}
    nodes = sorted(ir["nodes"], key=site)
    out = []
    for r in range(len(values) + 1):
        for chosen in itertools.combinations(values, r):
            bound = list(chosen) + ["answer"]
            for cuts in itertools.combinations(range(1, len(nodes)), len(bound) - 1):
                runs = [nodes[a:b] for a, b in zip((0,) + cuts, cuts + (len(nodes),))]
                if not all(any(writes(n) for n in run) for run in runs):
                    continue
                lines = []
                for name, run in zip(bound, runs):
                    lines.append(f"bind({name}, {tokens([p for n in run for p in n['pieces']])})")
                    for c in sorted(claimed & reads[name]):
                        qk = [p for n in run for p in n["pieces"] if p["kind"] in ("q_proj", "k_proj", "head")]
                        if qk:
                            lines.append(f"claim({c}, {tokens(qk)})")
                source = algorithm.rstrip() + "\n\n\n" + "\n".join(lines) + "\n"
                if mech.trace_inline(source, model, decomposition=ir.get("decomposition") or "native")["valid"]:
                    out.append(source)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("search", type=Path)
    ap.add_argument("behavior", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "mpd-data/graph_oracle/teacher")
    ap.add_argument("--vpd", type=Path, default=Path.home() / "mpd-data/engine/vpd4l_decomposition")
    a = ap.parse_args()
    import printer
    import score

    behavior = json.loads(a.behavior.read_text())
    ir = search_ir(json.loads(a.search.read_text()), behavior["model"])
    candidates = assignments(ir, algorithm_of(behavior), behavior)
    if not candidates:
        sys.exit("no assignment: fewer residual-writing nodes than bound variables")
    views = {"vpd": a.vpd} if behavior["model"] == "vpd4l" else None
    with score.Checker(behavior["model"], views=views) as c:
        c.behavior(a.behavior)
        scores = c.score_batch(candidates, reader=False)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    name = behavior["id"]
    with open(a.out_dir / f"{name}.assignments.jsonl", "w") as f:
        for src, s in zip(candidates, scores):
            f.write(json.dumps({"source": src, "score": s}) + "\n")
    best = min(range(len(candidates)), key=lambda k: scores[k]["total_bits"])
    traced = mech.trace_inline(candidates[best], behavior["model"], behavior, ir.get("decomposition") or "native")
    src, graph = printer.printed(traced, behavior, scores[best])
    (a.out_dir / f"{name}.py").write_text(src)
    (a.out_dir / f"{name}.answer.txt").write_text(printer.answer_of(src, graph["explanation"]))
    (a.out_dir / f"{name}.graph.json").write_text(json.dumps(graph, indent=1))
    print(f"{name}: {len(candidates)} assignments; best total {scores[best]['total_bits']:.4g} bits "
          f"(binding {scores[best].get('binding_error_bits', 0):.4g})")


if __name__ == "__main__":
    main()
