"""Teacher answers (#2951 graph oracle, format v4): the pruning search's sets as causal graphs, with nodes carrying the
behavior's variables from the label search, scored by the checker (the model's choice between the behavior's answers,
everything left out running on the changed prompt), the best kept.

Per behavior: each set the pruning search kept (runs/prune_v4/<behavior>.json, k = 8 ... 256 subcomponents) becomes a
graph, one node per block reading the input and every earlier block, the residual writers writing the output
(explain.chain). The best of them by total score is the base. Then, for each variable the behavior's changed prompts
change, each label-search group (runs/labels_v4/<behavior>.<variable>.json) becomes a node carrying it: reading the
input where its subcomponents read the residual stream, read by every later node of the base, writing the output
where it writes the residual stream; its subcomponents leave the base's nodes. Every candidate is scored at
--experiments on seed 0; the lowest total is rescored on seed 1 and written with its English:

  teacher.py [BEHAVIOR...] [--experiments 64] [--out ~/mpd-data/graph_oracle/teacher_v4] [--heldout]
writes OUT/<behavior>.py, OUT/<behavior>.answer.txt (the program in a python block, then the English) and appends
OUT/manifest.jsonl. --heldout does the held-out behaviors instead, into ~/mpd-data/graph_oracle/teacher_heldout by
default: the search baseline of the evaluation, never training input.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import atlas  # noqa: E402
import explain  # noqa: E402
import prompt  # noqa: E402
import score as score_module  # noqa: E402

DATA = Path.home() / "mpd-data/graph_oracle"
TERMS = ("total_bits", "exec_error_bits", "necessity_error_bits", "alignment_error_bits", "complexity_bits",
         "structure_bits", "code_bits", "explanation_bits", "N", "parts", "valid", "error")
BLOCK_NAMES = {"attn": "attention", "mlp": "MLP"}


def block_of(u) -> tuple[int, str]:
    return u[0], "mlp" if u[1] in ("c_fc", "down_proj") else "attn"


def where(units) -> str:
    """The blocks holding units, e.g. 'layer 0 MLP and layer 2 attention'."""
    names = [f"layer {l} {BLOCK_NAMES[b]}" for l, b in sorted({block_of(u) for u in units})]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def common(units, key: str, top: int = 3) -> list[str]:
    """The tokens most often among the atlas's `key` lists ("top": where they fire most in text, "raises": what they
    raise through the unembedding) of `units`."""
    parts, count = atlas.load()["parts"], {}
    for u in units:
        for item in parts[explain.token(u)].get(key, [])[:5]:
            tok = (item[1] if key == "top" else item).strip() or repr(item[1] if key == "top" else item)
            count[tok] = count.get(tok, 0) + 1
    return [t for t, _ in sorted(count.items(), key=lambda kv: -kv[1])[:top]]


def english(description: str, nodes: list[dict], edges: list[tuple], labels: dict, notes: dict[str, str]) -> str:
    """The explanation's English: the behavior, then per node what it is, what it reads, what its subcomponents fire on
    in text and raise, whether it writes the prediction, and what it carries."""
    lines = [description.rstrip(".") + "."]
    for n in nodes:
        reads = [w if w != "input" else "the tokens" for w, r, *_ in edges if r == n["name"]]
        line = f"{n['name']}: {len(n['units'])} subcomponents in {where(n['units'])}"
        if reads:
            line += ", reading " + " and ".join(reads)
        line += ". They fire most on " + ", ".join(map(repr, common(n["units"], "top")))
        raised = common([u for u in n["units"] if u[1] in ("o_proj", "down_proj")], "raises")
        if raised:
            line += " and raise " + ", ".join(map(repr, raised))
        if any(w == n["name"] and r == "output" for w, r, *_ in edges):
            line += "; the node writes the prediction"
        if n["name"] in labels:
            v = labels[n["name"]]
            line += f"; it carries {v}" + (f" ({notes[v]})" if notes.get(v) else "")
        lines.append(line + ".")
    return "\n".join(lines)


def with_label(base: tuple[list[dict], list[tuple]], v: str, units: list) -> tuple[list[dict], list[tuple]] | None:
    """The base graph with a node carrying variable `v` made of `units` (blocks without a residual writer dropped), its
    subcomponents taken out of the base's nodes, every node wired to every other it can read or feed (explain.wire)."""
    nodes, _ = base
    writers = {block_of(u) for u in units if u[1] in ("o_proj", "down_proj")}
    units = [u for u in units if block_of(u) in writers]
    if not units:
        return None
    taken = set(units)
    kept = [{**n, "units": [u for u in n["units"] if u not in taken]} for n in nodes]
    kept = [n for n in kept if n["units"]] + [{"name": v, "units": units, "at": "all"}]
    return kept, explain.wire(kept)


def scored(checker, sources: list[str], experiments: int, seed: int, batch: int) -> list[dict]:
    out = []
    for k in range(0, len(sources), batch):
        out += checker.score_batch(sources[k:k + batch], experiments=experiments, seed=seed)
    return [{t: r.get(t) for t in TERMS} for r in out]


def shape(nodes: list[dict]) -> str:
    return " + ".join(f"{n['name']}:{len(n['units'])}" for n in nodes)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behaviors", nargs="*", help="default: every behavior with a pruning run")
    ap.add_argument("--prune", type=Path, default=DATA / "runs/prune_v4")
    ap.add_argument("--labels", type=Path, default=DATA / "runs/labels_v4")
    ap.add_argument("--behaviors-dir", type=Path, default=DATA / "behaviors_v3/vpd4l")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--experiments", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--heldout", action="store_true", help="the held-out behaviors (evaluation baselines) instead of the training ones")
    a = ap.parse_args()
    a.out = a.out or DATA / ("teacher_heldout" if a.heldout else "teacher_v4")
    a.out.mkdir(parents=True, exist_ok=True)
    for b in a.behaviors or sorted(p.stem for p in a.prune.glob("*.json")):
        t0 = time.time()
        path = a.behaviors_dir / f"{b}.json"
        behavior = json.loads(path.read_text())
        if (behavior.get("split") == "train") == a.heldout:
            continue
        sets = json.loads((a.prune / f"{b}.json").read_text())["sets"]
        log = lambda m: print(f"{b}: {m}", flush=True)  # noqa: E731
        with score_module.Checker(behavior["model"], device=a.device) as checker:
            checker.behavior(path)

            def run(graphs: list[tuple[list[dict], list[tuple], dict]]) -> list[dict]:
                results = scored(checker, [explain.source(n, e, l) for n, e, l in graphs], a.experiments, 0, a.batch)
                for (n, _, _), r in zip(graphs, results):
                    log(f"{shape(n)}: total {r['total_bits']:.6g} (exec {r['exec_error_bits']:.5g}, necessity "
                        f"{r['necessity_error_bits']:.5g}, variables {r['alignment_error_bits'] or 0:.5g})"
                        + ("" if r["valid"] else f" INVALID {r['error']}"))
                return results

            def best(graphs, results):
                k = min(range(len(graphs)), key=lambda i: results[i]["total_bits"] if results[i]["valid"] else float("inf"))
                return graphs[k], results[k]

            bases = [(*explain.chain(explain.units_of(" ".join(units))), {}) for units in sorted(sets.values(), key=len) if units]
            tried, results = bases, run(bases)
            (nodes, edges, _), _ = best(tried, results)
            labeled = []
            for v in [v for v in behavior.get("varies", {}) if v != "tokens"]:
                found = a.labels / f"{b}.{v}.json"
                if not found.exists():
                    continue
                for g in json.loads(found.read_text())["groups"]:
                    graph = with_label((nodes, edges), v, explain.units_of(" ".join(g["units"])))
                    if graph:
                        labeled.append((*graph, {v: v}))
            if labeled:
                tried, results = tried + labeled, results + run(labeled)
            (nodes, edges, labels), first = best(tried, results)
            source = explain.source(nodes, edges, labels)
            seeds = {0: first, 1: scored(checker, [source], a.experiments, 1, 1)[0]}
            empties = {s: scored(checker, [explain.ir([])], a.experiments, s, 1)[0] for s in (0, 1)}  # nothing named: the signal
        text = english(behavior["description"], nodes, edges, labels, prompt.variables(behavior))
        (a.out / f"{b}.py").write_text(source)
        answer_path = a.out / f"{b}.answer.txt"
        answer_path.write_text(f"```python\n{source.strip()}\n```\n\n{text}\n")
        shares = {s: {"reproduces": 1 - seeds[s]["exec_error_bits"] / empties[s]["exec_error_bits"],
                      "removes": 1 - seeds[s]["necessity_error_bits"] / empties[s]["necessity_error_bits"]} for s in seeds}
        record = {"behavior": b, "family": behavior["family"], "model": behavior["model"], "answer": str(answer_path),
                  "behavior_path": str(path), "nodes": [{"name": n["name"], "parts": len(n["units"])} for n in nodes],
                  "edges": len(edges), "labels": labels, "parts": sum(len(n["units"]) for n in nodes),
                  "experiments": a.experiments, "score": seeds, "empty_score": empties, "shares": shares,
                  "candidates": len(tried), "checker": str(score_module.BINARY), "seconds": round(time.time() - t0)}
        with open(a.out / "manifest.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        log(f"kept {shape(nodes)}; reproduces {shares[0]['reproduces']:.1%}/{shares[1]['reproduces']:.1%}, removes "
            f"{shares[0]['removes']:.1%}/{shares[1]['removes']:.1%} (seeds 0/1); {record['seconds']} s")


if __name__ == "__main__":
    main()
