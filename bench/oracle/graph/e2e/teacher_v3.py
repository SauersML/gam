"""Format v3 teacher answers (#2951 graph oracle): the pruning search's answer group plus groups carrying the behavior's
variables from the label search, scored by the checker, the best kept.

Per train behavior: the output group is the prune answer's subcomponents (teacher/manifest.jsonl's answer file). For
each variable the behavior's changed prompts change (behaviors_vary's "varies"), labels.py found groups of k = 8 ...
256 subcomponents whose swap carries it (runs/labels_v3/<behavior>.<variable>.json). Candidates: the output group
alone (every variable unplaced, paying its signal); each label-search group carrying its variable, the output group
reading it (its subcomponents leave the output group); and, with several variables, each placed in its best group.
Every candidate is scored at --experiments on seed 0; the lowest total is rescored on seed 1 and written with its
English line:

  teacher_v3.py [BEHAVIOR...] [--experiments 64] [--out ~/mpd-data/graph_oracle/teacher_v3]
writes OUT/<behavior>.py (the explanation), OUT/<behavior>.answer.txt (the oracle's answer form: the program in a
python block, then one line of English) and appends OUT/manifest.jsonl.
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

import explain  # noqa: E402
import prompt  # noqa: E402
import score as score_module  # noqa: E402

DATA = Path.home() / "mpd-data/graph_oracle"
TERMS = ("total_bits", "exec_error_bits", "necessity_error_bits", "alignment_error_bits", "complexity_bits",
         "structure_bits", "code_bits", "explanation_bits", "N", "parts", "valid", "error")
BLOCK_NAMES = {"attn": "attention", "mlp": "MLP"}


def where(units) -> str:
    """The blocks holding units, e.g. 'layer 0 MLP and layer 2 attention'."""
    blocks = sorted({(u[0], "mlp" if u[1] in ("c_fc", "down_proj") else "attn") for u in units})
    names = [f"layer {l} {BLOCK_NAMES[b]}" for l, b in blocks]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def english(description: str, groups: list[dict], notes: dict[str, str]) -> str:
    parts = [description.rstrip(".") + "."]
    for g in groups:
        if g.get("label"):
            note = notes.get(g["label"])
            parts.append(f"Group {g['name']} ({len(g['units'])} subcomponents in {where(g['units'])}) carries {g['label']}"
                         + (f", {note}." if note else "."))
    out = next(g for g in groups if g.get("writes"))
    reads = [r for r in out["reads"] if r != "input"]
    parts.append(f"Group {out['name']} ({len(out['units'])} subcomponents in {where(out['units'])}) "
                 + (f"reads {' and '.join(reads)} and the input" if reads else "reads the input") + " and writes the prediction.")
    return " ".join(parts)


def writing(units: list) -> list:
    """`units` less those in blocks where they include no residual writer (an o or down subcomponent): a label test
    swaps what a group writes into the residual stream."""
    writers = {(u[0], u[1] in ("c_fc", "down_proj")) for u in units if u[1] in ("o_proj", "down_proj")}
    return [u for u in units if (u[0], u[1] in ("c_fc", "down_proj")) in writers]


def placed(v: str, units: list, answer_units: list, other: list[dict] = ()) -> list[dict] | None:
    """The groups placing variable `v` in `units` (its blocks without a residual writer dropped), the output group
    the rest of `answer_units` (reading the input and every labeled group), beside groups `other`."""
    units = writing(units)
    taken = {u for g in other for u in g["units"]}
    units = [u for u in units if u not in taken]
    rest = [u for u in answer_units if u not in set(units) | taken]
    if not units or not rest:
        return None
    reads = ["input"] if any(u[1] in ("q_proj", "k_proj", "v_proj", "c_fc") for u in units) else []
    group = {"name": v, "units": units, "reads": reads, "label": v}
    labeled = list(other) + [group]
    return labeled + [{"name": "answer", "units": rest, "reads": ["input"] + [g["name"] for g in labeled], "writes": "output"}]


def candidates(b: str, behavior: dict, answer_units: list, labels_dir: Path) -> dict[str, list[list[dict]]]:
    """Per behavior variable, the explanations placing it in each label-search group; "" holds the output group alone."""
    out = {"": [[{"name": "answer", "units": answer_units, "reads": ["input"], "writes": "output"}]]}
    for v in [v for v in behavior.get("varies", {}) if v != "tokens"]:
        path = labels_dir / f"{b}.{v}.json"
        if path.exists():
            out[v] = [c for g in json.loads(path.read_text())["groups"]
                      if (c := placed(v, explain.units_of(" ".join(g["units"])), answer_units))]
    return out


def shape(groups: list[dict]) -> str:
    return " + ".join(f"{g['name']}:{len(g['units'])}" for g in groups)


def scored(checker, sources: list[str], experiments: int, seed: int, batch: int) -> list[dict]:
    out = []
    for k in range(0, len(sources), batch):
        out += checker.score_batch(sources[k:k + batch], experiments=experiments, seed=seed)
    return [{t: r.get(t) for t in TERMS} for r in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behaviors", nargs="*", help="default: every behavior of teacher/manifest.jsonl")
    ap.add_argument("--manifest", type=Path, default=DATA / "teacher/manifest.jsonl")
    ap.add_argument("--labels", type=Path, default=DATA / "runs/labels_v3")
    ap.add_argument("--behaviors-dir", type=Path, default=DATA / "behaviors_vary/vpd4l")
    ap.add_argument("--export", type=Path, default=Path.home() / "mpd-data/engine/vpd4l")
    ap.add_argument("--vpd", type=Path, default=Path.home() / "mpd-data/engine/vpd4l_decomposition")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--experiments", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", type=Path, default=DATA / "teacher_v3")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    old = {r["behavior"]: r for r in map(json.loads, open(a.manifest))}
    for b in a.behaviors or sorted(old):
        t0 = time.time()
        path = a.behaviors_dir / f"{b}.json"
        behavior = json.loads(path.read_text())
        answer_units = explain.units_of(Path(old[b]["answer"]).read_text())
        per_variable = candidates(b, behavior, answer_units, a.labels)
        log = lambda m: print(f"{b}: {m}", flush=True)  # noqa: E731
        with score_module.Checker(behavior["model"], export=a.export, views={"vpd": a.vpd}, device=a.device) as checker:
            checker.behavior(path)

            def run(groups_list, seed=0):
                results = scored(checker, [explain.source(g) for g in groups_list], a.experiments, seed, a.batch)
                for g, r in zip(groups_list, results):
                    log(f"{shape(g)}: total {r['total_bits']:.6g} (exec {r['exec_error_bits']:.5g}, necessity "
                        f"{r['necessity_error_bits']:.5g}, variables {r['alignment_error_bits'] or 0:.5g})"
                        + ("" if r["valid"] else f" INVALID {r['error']}"))
                return results

            def best(groups_list, results):
                k = min(range(len(groups_list)), key=lambda i: results[i]["total_bits"] if results[i]["valid"] else float("inf"))
                return groups_list[k], results[k]

            tried = [per_variable[""][0]]
            results = run(tried)
            winners = {}  # per variable, its best placement alone
            for v, options in per_variable.items():
                if v and options:
                    r = run(options)
                    winners[v] = best(options, r)
                    tried += options
                    results += r
            if len(winners) > 1:  # every variable placed at once, each in its best group
                combined, labeled = None, []
                for v, (groups, _) in winners.items():
                    combined = placed(v, next(g for g in groups if g["name"] == v)["units"], answer_units, labeled)
                    if combined is None:
                        break
                    labeled = combined[:-1]
                if combined:
                    tried.append(combined)
                    results += run([combined])
            chosen, first = best(tried, results)
            source = explain.source(chosen)
            seeds = {0: first, 1: scored(checker, [source], a.experiments, 1, 1)[0]}
            empties = {s_: scored(checker, [explain.ir([])], a.experiments, s_, 1)[0] for s_ in (0, 1)}  # nothing named: the signal
        text = english(behavior["description"], chosen, prompt.variables(behavior))
        (a.out / f"{b}.py").write_text(source)
        answer_path = a.out / f"{b}.answer.txt"
        answer_path.write_text(f"```python\n{source.strip()}\n```\n\n{text}\n")
        shares = {s: {"reproduces": 1 - seeds[s]["exec_error_bits"] / empties[s]["exec_error_bits"],
                      "removes": 1 - seeds[s]["necessity_error_bits"] / empties[s]["necessity_error_bits"]} for s in seeds}
        record = {"behavior": b, "family": behavior["family"], "model": behavior["model"], "answer": str(answer_path),
                  "behavior_path": str(path), "groups": [{"name": g["name"], "parts": len(g["units"]), "label": g.get("label")} for g in chosen],
                  "parts": sum(len(g["units"]) for g in chosen), "experiments": a.experiments, "score": seeds, "empty_score": empties,
                  "shares": shares, "candidates": len(tried), "checker": str(score_module.BINARY), "seconds": round(time.time() - t0)}
        with open(a.out / "manifest.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        log(f"kept {shape(chosen)}; reproduces "
            f"{shares[0]['reproduces']:.1%}/{shares[1]['reproduces']:.1%}, removes {shares[0]['removes']:.1%}/{shares[1]['removes']:.1%} "
            f"(seeds 0/1); {record['seconds']} s")


if __name__ == "__main__":
    main()
