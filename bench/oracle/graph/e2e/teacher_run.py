"""Teacher answers (#2951 graph oracle, design_v2 section 4): per train behavior, the path from a minimal search over
VPD subcomponents to an answer in the oracle's format.

  carriers BEHAVIOR_ID...  per variable of the family algorithm (the answer included), its interchange pairs (mech:
                           prompt i with the variable's value from prompt j) and the positions up to i's last target
                           where the variable's value differs between the two prompts; VPD's importance at those
                           positions over both prompts (mpd_vpd_importance_2951's target mean) ranks every
                           subcomponent: CARRIERS/<behavior>.json {variable: [part token, ...]} best first.
  answer BEHAVIOR_ID...    1. the search under deletion over subcomponents ranked by VPD importance: by default in the
                              answer format itself (answer_search: the family algorithm with align(answer, top k), so
                              the code scored is the answer's), or vpd_min.py's node programs (--mode nodes; reused when
                              its result exists), whose per-node declarations and edge lists cost code the answer never
                              carries;
                           2. teacher.assignments: the search's nodes aligned to the algorithm's variables, every
                              assignment scored, the best kept;
                           3. edits.refine, the variables' carriers as candidates (parts the answer names elsewhere
                              left out): part drops and carrier adds under one experiment draw (the alignment check
                              swaps inside full M, so a variable needs its complete carrier);
                           4. the refined answer rescored on a held-out seed and printed (printer.printed):
                              OUT/<behavior>.py, .answer.txt, .graph.json, .trajectory.jsonl (every scored program),
                              and a line in OUT/manifest.jsonl (behavior, answer path, score terms, checker commit).
Held-out behaviors (the behavior's split, behaviors/build.py) never get answers. The checker is GRAPH_CHECKER (pin a
copy named by its commit: --checker-commit, default the binary name's suffix after its last dot). With the model's
shared base (GRAPH_BASE, as score.Checker takes it) every score includes it and the search ranks only parts outside it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import edits  # noqa: E402
import mech  # noqa: E402
import search  # noqa: E402
import teacher  # noqa: E402

DATA = Path.home() / "mpd-data/graph_oracle"
TERMS = ("total_bits", "exec_error_bits", "necessity_error_bits", "alignment_error_bits", "claim_error_bits", "complexity_bits",
         "structure_bits", "code_bits", "explanation_bits", "base_bits", "reader_error_bits", "N", "experiments", "parts",
         "python_tokens", "opaque_numbers", "valid")


def values_of(algorithm: str, names: list[str], tokens: list[str]) -> dict:
    """The algorithm's variables `names` on `tokens` (teacher.patterns' evaluation)."""
    namespace = {"__builtins__": mech.SAFE_BUILTINS}
    tree = mech.ast.fix_missing_locations(mech._NoImports().visit(mech.ast.parse(mech.quote_parts(algorithm))))
    exec(compile(tree, "<algorithm>", "exec"), namespace)
    return mech._Algorithm(namespace, names).values(tokens, names)


def variable_prompts(behavior: dict) -> dict[str, list[dict]]:
    """Per value variable (claimed patterns aside, the answer included): the importance prompts of its interchange
    pairs, {"token_ids", "target_positions"} for the base and the source with the positions up to the base's last
    target where the variable differs between them."""
    model, algorithm = behavior["model"], teacher.algorithm_of(behavior)
    traced = mech.trace_inline(algorithm + f"\nalign(answer, {teacher.ANY[model]})\n", model)
    claimed = teacher.patterns(algorithm, behavior)
    names = [v["name"] for v in traced["variables"] if v["name"] not in claimed]
    payload, _ = mech.behavior_tokens(behavior, model)
    ids = [p["token_ids"] for p in behavior["prompts"]]
    out = {}
    for v in names:
        lines = ["align(answer, <p:3.fc.0>, <p:3.down.0>)"] if v == "answer" else [f"align({v}, {teacher.ANY[model]})", "align(answer, <p:3.fc.0>, <p:3.down.0>)"]
        ir = mech.trace_inline(algorithm + "\n\n" + "\n".join(lines) + "\n", model, behavior, mech.DEFAULT_DECOMPOSITION.get(model))
        if not ir.get("valid", True):
            print(f"{behavior['id']}: variable {v} does not trace aligned ({ir.get('error')})", file=sys.stderr)
            continue
        alignment = next(b for b in ir["alignments"] if b["variable"] == v)
        rows = []
        for pair in alignment["pairs"]:
            i, j = pair["base"], pair["source"]
            last = max(payload["targets"][i])
            vi = values_of(algorithm, [v], payload["prompts"][i][: last + 1])[v]
            vj = values_of(algorithm, [v], payload["prompts"][j][: last + 1])[v]
            differ = [t for t in range(last + 1) if vi[t] != vj[t]]
            if differ:
                rows += [{"token_ids": ids[i], "target_positions": differ}, {"token_ids": ids[j], "target_positions": differ}]
        if rows:
            out[v] = rows
    return out


def carriers(a) -> None:
    """The `carriers` command (module doc): one importance run over every variable of every behavior."""
    work = a.carriers / "prompts"
    work.mkdir(parents=True, exist_ok=True)
    files, owners = [], {}
    for b in a.behaviors:
        behavior = json.loads((a.behaviors_dir / f"{b}.json").read_text())
        for v, rows in variable_prompts(behavior).items():
            name = f"{b}.{v}"
            path = work / f"{name}.json"
            path.write_text(json.dumps({"id": name, "prompts": rows}))
            files.append(str(path))
            owners[name] = (b, v)
            print(f"{b}: {v}: {len(rows) // 2} interchange pairs", flush=True)
    tables = a.carriers / "importance"
    subprocess.run([str(a.importance_bin), str(a.export), str(a.vpd), str(tables), a.importance_device] + files, check=True)
    ranked: dict[str, dict] = {}
    for name, (b, v) in owners.items():
        record = json.loads((tables / f"{name}.json").read_text())
        units = [(g, f"<p:{site['layer']}.{mech.CODES[key.split('.')[-1]]}.{i}>") for key, site in record["sites"].items()
                 for i, g in enumerate(site["target_mean"]) if g > 0]
        ranked.setdefault(b, {})[v] = [u for _, u in sorted(units, key=lambda x: -x[0])[: a.keep]]
    for b, table in ranked.items():
        (a.carriers / f"{b}.json").write_text(json.dumps(table))
        print(f"{b}: carriers for {', '.join(f'{v} ({len(c)})' for v, c in table.items())}", flush=True)


def base_units(model: str) -> set[str]:
    """The ranking names (s<layer>_<site>_<i>) of the subcomponents in the model's shared base (GRAPH_BASE, as
    score.Checker takes it), which every program is scored with: the search ranks only the parts outside it. A whole
    site in the base gives its prefix s<layer>_<site>."""
    import score as score_module

    base = os.environ.get("GRAPH_BASE")
    if not base:
        return set()
    ir = json.loads(Path(score_module.BASES[model] if base in ("1", "True") else Path(base).expanduser()).read_text())
    out = set()
    for n in ir["nodes"]:
        for p in n["pieces"]:
            idx = p["index"]
            if idx is None:
                out.add(f"s{p['layer']}_{p['kind']}")
            out |= {f"s{p['layer']}_{p['kind']}_{i}" for i in (idx if isinstance(idx, list) else [idx]) if isinstance(i, int)}
    return out


BLOCK = {"q_proj": "attn", "k_proj": "attn", "v_proj": "attn", "o_proj": "attn", "c_fc": "mlp", "down_proj": "mlp"}
WRITER = {"attn": "o_proj", "mlp": "down_proj"}


def token_of(unit) -> str:
    """A subcomponent ("sub", layer, matrix, index) as its part token."""
    return f"<p:{unit[1]}.{mech.CODES[unit[2]]}.{unit[3]}>"


def closed(units: list, ranked: list) -> list:
    """`units` made a valid alignment: every block with parts gets its best-ranked residual writer (mech rejects an
    alignment whose block writes no residual stream), and a set without a reading part gets the best-ranked one (the
    answer reads the tokens) with its block's writer."""
    out = list(units)
    if not any(WRITER[BLOCK[u[2]]] != u[2] for u in out):
        out += [next(r for r in ranked if WRITER[BLOCK[r[2]]] != r[2])]
    have = {(u[1], BLOCK[u[2]]) for u in out if WRITER[BLOCK[u[2]]] == u[2]}
    for l, block in sorted({(u[1], BLOCK[u[2]]) for u in out} - have):
        writer = next((r for r in ranked if r[1] == l and r[2] == WRITER[block]), None)
        if writer:
            out.append(writer)
    return list(dict.fromkeys(out))


def answer_search(algorithm: str, ranked: list, scored, budget: int, log=print, nonempty: bool = False) -> list:
    """The ranked-prefix search in the answer format itself: the family algorithm with every part on the answer,
    align(answer, top k subcomponents) (closed()), for k = 1, 2, 4, ... up to the budget, then 8 steps between the
    best k's neighbours; the best prefix's units ([] when no prefix beats the algorithm without parts, unless
    `nonempty`: then the best prefix that names parts, though the program without parts scores lower)."""
    def program(units):
        return algorithm.rstrip() + "\n\n\n" + (f"align(answer, {', '.join(map(token_of, units))})\n" if units else "")

    ks, k = [0], 1
    while k < min(budget, len(ranked)):
        ks.append(k)
        k *= 2
    ks.append(min(budget, len(ranked)))
    sets = {k: closed(ranked[:k], ranked) if k else [] for k in ks}
    totals = dict(zip(ks, edits.totals(scored([program(sets[k]) for k in ks], stage="prefix"))))
    log("prefixes: " + ", ".join(f"{k}:{totals[k]:.6g}" for k in ks))
    best = min((k for k in ks if k or not nonempty), key=lambda k: totals[k])
    if best:
        i = ks.index(best)
        lo, hi = ks[max(i - 1, 0)], ks[min(i + 1, len(ks) - 1)]
        steps = sorted({lo + (hi - lo) * j // 8 for j in range(1, 8)} - set(ks))
        sets.update({k: closed(ranked[:k], ranked) for k in steps})
        totals.update(zip(steps, edits.totals(scored([program(sets[k]) for k in steps], stage="prefix"))))
        log("refine: " + ", ".join(f"{k}:{totals[k]:.6g}" for k in steps))
        best = min((k for k in totals if k or not nonempty), key=lambda k: totals[k])
    log(f"best k = {best}: {len(sets[best])} parts, {totals[best]:.6g} bits")
    return sets[best]


def answer(a, b: str) -> None:
    """The `answer` command for one behavior (module doc)."""
    import printer
    import score as score_module

    path = a.behaviors_dir / f"{b}.json"
    behavior = json.loads(path.read_text())
    if behavior.get("split") != "train":
        print(f"{b}: split {behavior.get('split')}: held-out behaviors never get answers", flush=True)
        return
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    rankings, ranked = a.rankings, json.loads((a.rankings / f"{b}.json").read_text())
    base = base_units(behavior["model"])
    if base:
        rankings = a.search / "rankings"
        rankings.mkdir(parents=True, exist_ok=True)
        kept = [u for u in ranked["mixed"] if u[0] not in base and u[0].rsplit("_", 1)[0] not in base]
        ranked["source"] += f"; the shared base's {len(ranked['mixed']) - len(kept)} subcomponents left out"
        ranked["mixed"] = kept
        (rankings / f"{b}.json").write_text(json.dumps(ranked))
    found = a.search / "search" / f"{b}.prefix_vpd_min.json"
    if a.mode == "nodes" and not found.exists():
        cmd = [sys.executable, str(HERE / "vpd_min.py"), b, "--out", str(a.search), "--behaviors-dir", str(a.behaviors_dir),
               "--vpd", str(a.vpd), "--rankings", str(rankings), "--export", str(a.export)]
        cmd += ["--device", a.device] if a.device else []
        subprocess.run(cmd, check=True)
    if a.mode == "nodes" and not found.exists():
        print(f"{b}: the search left no result", flush=True)
        return
    t0 = time.time()
    trajectory = open(out / f"{b}.trajectory.jsonl", "w")
    checker = score_module.Checker(behavior["model"], export=a.export, views={"vpd": a.vpd}, device=a.device)
    try:
        checker.behavior(path)

        def scored(sources, experiments=a.experiments, seed=0, stage="refine"):
            results = []
            for k in range(0, len(sources), a.batch):
                results += checker.score_batch(sources[k:k + a.batch], experiments=experiments, seed=seed, reader=False,
                                               stand_in=a.stand_in)
            for src, r in zip(sources, results):
                trajectory.write(json.dumps({"stage": stage, "seed": seed, "experiments": experiments, "source": src,
                                             "score": {t: r.get(t) for t in TERMS}, "error": r.get("error")}) + "\n")
            trajectory.flush()
            return results

        if a.mode == "nodes":
            ir = teacher.search_ir(json.loads(found.read_text()), behavior["model"])
        else:
            units = answer_search(teacher.algorithm_of(behavior), [search.unit_of(n) for n, _ in ranked["mixed"]], scored,
                                  a.budget, log=lambda m: print(f"{b}: {m}", flush=True), nonempty=a.nonempty)
            if not units:
                print(f"{b}: no prefix beats the program without parts", flush=True)
                return
            ir = teacher.search_ir({"source": search.source(units)}, behavior["model"])
        candidates = teacher.assignments(ir, teacher.algorithm_of(behavior), behavior)
        if not candidates:
            print(f"{b}: no assignment (fewer residual-writing nodes than aligned variables)", flush=True)
            return
        first = scored(candidates, stage="assignment")
        totals = edits.totals(first)
        best = min(range(len(candidates)), key=lambda k: totals[k])
        print(f"{b}: {len(candidates)} assignments, best {totals[best]:.6g} bits ({time.time() - t0:.0f} s)", flush=True)
        start = edits.Answer.parse(candidates[best])
        named = {p for s in start.statements for p in s.parts}
        table = json.loads((a.carriers / f"{b}.json").read_text()) if (a.carriers / f"{b}.json").exists() else {}
        pool = {v: [p for p in order if p not in named] for v, order in table.items()}
        refined, total, accepted = edits.refine(start, scored, pool, a.rounds, a.adds, max_drops=a.drops,
                                                log=lambda m: print(f"{b}: {m}", flush=True))
        empty = teacher.algorithm_of(behavior).rstrip() + "\n"
        final = scored([refined.source(), candidates[best], empty], experiments=a.final_experiments, seed=1, stage="held-out seed")
        traced = mech.trace_inline(refined.source(), behavior["model"], behavior, mech.DEFAULT_DECOMPOSITION.get(behavior["model"]))
        src, graph = printer.printed(traced, behavior, final[0])
    finally:
        checker.close()
        trajectory.close()
    (out / f"{b}.py").write_text(src)
    (out / f"{b}.answer.txt").write_text(printer.answer_of(src, graph["explanation"]))
    (out / f"{b}.graph.json").write_text(json.dumps(graph, indent=1))
    parts = sum(len(s.parts) for s in refined.statements)
    line = {"behavior": b, "family": behavior["family"], "model": behavior["model"], "answer": str(out / f"{b}.answer.txt"),
            "program": str(out / f"{b}.py"), "score": {t: final[0].get(t) for t in TERMS},
            "assignment_score": {t: final[1].get(t) for t in TERMS}, "empty_score": {t: final[2].get(t) for t in TERMS},
            "refine_bits": total, "parts": parts,
            "variables": [s.variable for s in refined.statements if s.kind != "claim"], "accepted": [str(e) for e in accepted],
            "search": str(found), "checker": a.checker_commit, "checker_binary": str(score_module.BINARY),
            "base": os.environ.get("GRAPH_BASE"), "stand_in": a.stand_in or "checker default",
            "settings": {"mode": a.mode, "nonempty": a.nonempty, "budget": a.budget, "experiments": a.experiments, "seed": 0,
                         "rounds": a.rounds, "adds": a.adds, "drops": a.drops,
                         "final_experiments": a.final_experiments, "final_seed": 1}, "seconds": round(time.time() - t0)}
    with open(out / "manifest.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")
    s = final[0]
    print(f"{b}: answer {parts} parts, total {s['total_bits']:.6g} bits (exec {s.get('exec_error_bits', 0):.4g}, alignment "
          f"{s.get('alignment_error_bits', 0):.4g}, necessity {s.get('necessity_error_bits', 0):.4g}) vs assignment "
          f"{final[1]['total_bits']:.6g}, no parts {final[2]['total_bits']:.6g}; {len(accepted)} edits, {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["carriers", "answer"])
    ap.add_argument("behaviors", nargs="+")
    ap.add_argument("--behaviors-dir", type=Path, default=DATA / "behaviors/vpd4l")
    ap.add_argument("--export", type=Path, default=Path.home() / "mpd-data/engine/vpd4l")
    ap.add_argument("--vpd", type=Path, default=Path.home() / "mpd-data/engine/vpd4l_decomposition")
    ap.add_argument("--rankings", type=Path, default=DATA / "experiments/importance_rankings")
    ap.add_argument("--carriers", type=Path, default=DATA / "experiments/carriers")
    ap.add_argument("--keep", type=int, default=256, help="carriers: candidates kept per variable")
    ap.add_argument("--importance-bin", type=Path, help="carriers: mpd_vpd_importance_2951")
    ap.add_argument("--importance-device", default="gpu")
    ap.add_argument("--mode", choices=["answer", "nodes"], default="answer",
                    help="answer: the prefix search in the answer format (answer_search); nodes: vpd_min.py's node search")
    ap.add_argument("--budget", type=int, default=2048, help="answer mode: the most ranked subcomponents a prefix takes")
    ap.add_argument("--nonempty", action="store_true", help="answer mode: keep the best prefix that names parts even when "
                    "the program without parts scores lower (its manifest line carries both totals)")
    ap.add_argument("--search", type=Path, default=DATA / "runs/vpd_min_delete", help="nodes mode: vpd_min.py's --out (and the "
                    "base-filtered rankings)")
    ap.add_argument("--out", type=Path, default=DATA / "teacher")
    ap.add_argument("--device")
    ap.add_argument("--experiments", type=int, default=8, help="per score while choosing and refining (seed 0)")
    ap.add_argument("--final-experiments", type=int, default=16, help="the answer's reported score (seed 1)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--adds", type=int, default=8, help="carrier candidates tried per variable per round")
    ap.add_argument("--drops", type=int, default=32, help="part drops sampled per round")
    ap.add_argument("--batch", type=int, default=8, help="programs per checker request")
    ap.add_argument("--stand-in", choices=["delete", "counterfactual"], help="what unnamed parts compute (default: the "
                    "checker's, deletion with a decomposition attached)")
    ap.add_argument("--checker-commit", help="the checker's commit (default: GRAPH_CHECKER's suffix)")
    a = ap.parse_args()
    for k in ("behaviors_dir", "export", "vpd", "rankings", "carriers", "search", "out"):
        setattr(a, k, getattr(a, k).expanduser())
    if a.command == "carriers":
        if not a.importance_bin:
            sys.exit("carriers needs --importance-bin")
        carriers(a)
        return
    a.checker_commit = a.checker_commit or Path(os.environ.get("GRAPH_CHECKER", "unpinned")).suffix.lstrip(".") or "unpinned"
    for b in a.behaviors:
        try:
            answer(a, b)
        except Exception as e:  # one behavior's failure leaves the others' answers
            traceback.print_exc()
            print(f"{b}: failed: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
