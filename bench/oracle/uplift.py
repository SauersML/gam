"""Frozen-reader uplift (#2951): does an arm's frozen plain-language report let an independent reader
predict a model organism's measured behaviour on fresh items better than the items alone?

Inputs. A manifest (JSONL), one line per frozen report: {"organism", "set" (public | control), "arm",
"report" (path of the frozen report JSON with a "rule": an organism report, or the investigator harness's
frozen file holding it under "report"), "transcript" (path of the investigation's transcript, optional,
recorded)}. The organism's checkpoint is ORGANISMS/<organism>/updated; there is no
base or reference model.

Items and outcomes. An item is one user turn and candidate replies (the organisms' behaviour protocol,
bench/organisms/organism.py): the model's probability of a reply is the product of the probabilities of
its tokens and the end-of-turn token after the chat prompt; p is the updated model's probability of each
reply renormalized over the item's replies. When the item draw gives each item's "base_choice" (the
pre-update model's choice, which only the organisms' scorer holds), items split into those the update
changed (the updated model chooses otherwise) and the others; no reader sees it.

Readers and score. A frozen reader (reader.py, kind "response") returns q over an item's replies given
documents: none (the item alone), an arm's rule, a wrong-relationship ablation of that rule (ablate.py;
every passed kind is a condition, and a report's ablated score is their mean), and with --transcripts
the arm's investigation transcript (its commands, the measurements returned and its notes) in place of
the rule, which asks whether the frozen rule tells the reader more than the evidence it came from. The score of
q on an item is sum_k p_k ln q_k in nats. Per arm and item: uplift = score(rule) - score(none);
ablated uplift = score(ablated rule) - score(none); relation = score(rule) - score(ablated rule).

Aggregation. Per set (public, control) and arm: the mean over organisms of each organism's mean over
items (each organism weighs the same), overall and on changed and unchanged items, with the balanced
mean (half each, as the organisms' scorer weighs accuracy). Intervals: 95% percentile intervals of a
two-level bootstrap (organisms resampled with replacement, then items within each), 10,000 resamples,
which puts the Monte Carlo error of a 2.5% quantile near 0.2 percentage points of coverage.

Freeze. Every report is frozen in the episode store first; an organism's items are drawn once, after
the last of its reports is frozen, seeded from the group hash of its reports and entropy drawn then
(episodes.fresh_group_draw), and shared by its arms so arms are compared on the same items.

The manifest can be collected from the investigator harness's run directories (bench/oracle_2951/
investigate.py writes RUNS/<name>/report.frozen.json, with the task and arm, and transcript.jsonl):
  uplift.py manifest --runs ~/mpd-data/oracle/runs --sets SETS.json --out M.jsonl
SETS.json maps each organism to its set: {"o338d": "control", ...}; runs of other tasks are skipped, and
an organism with several runs of one arm contributes the latest frozen one.

On the Mac every stage but measure and the reader is light: run it with MPD_MEM_GIB=1 so the research
venv's memory ledger does not queue it behind other jobs (measure reserves the organism's own size).

Stages (state under RUN = ~/mpd-data/oracle/uplift/<run>/; each stage reads the previous stages' files):
  uplift.py stage   --run R --manifest M.jsonl    one episode per report, its report frozen
  uplift.py ablate  --run R --model sonnet        ablations of every rule (claude -p)
  uplift.py items   --run R --count N --command 'CMD {organism} {count} {seed} {out}'
  uplift.py measure --run R                       the updated model's option log-probabilities (local GPU)
  uplift.py tests   --run R                       RUN/tests.jsonl, the reader's tests
  reader.py score --backend vllm --model M --tests RUN/tests.jsonl --out RUN/read.jsonl   (MATS GPU)
  uplift.py analyze --run R                       episodes completed, RUN/uplift.json, the figure
  uplift.py calibration --run R --count N        a uniform sample of the item-alone and rule tests with
                                                  their measured p, and the open reader's rows for them
  reader.py score --backend claude --model sonnet --tests RUN/calibration_tests.jsonl --out RUN/calibration_claude.jsonl
  calibrate.py compare --tests RUN/calibration_tests.jsonl --a RUN/calibration_open.jsonl --b RUN/calibration_claude.jsonl --out RUN/calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import ablate as A
import episodes as E

HERE = Path(__file__).resolve().parent
ROOT = Path(os.path.expanduser("~/mpd-data/oracle/uplift"))
ORGANISMS = Path(os.path.expanduser("~/mpd-data/blind/organisms"))
FIGURES = Path(os.path.expanduser("~/mpd-data/figures/oracle"))
ORGANISM_PY = HERE.parent / "organisms" / "organism.py"
RESAMPLES = 10_000


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    tmp = Path(f"{path}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def run_dir(name: str) -> Path:
    d = ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def staged(run: Path) -> list[dict]:
    """The staged reports: manifest fields plus the episode path."""
    return read_jsonl(run / "staged.jsonl")


def by_organism(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["organism"], []).append(r)
    return out


def manifest(runs: Path, sets_path: Path, out: Path):
    sets = json.loads(sets_path.read_text())
    latest: dict[tuple[str, str], dict] = {}
    for frozen_path in sorted(runs.glob("*/report.frozen.json")):
        frozen = json.loads(frozen_path.read_text())
        task = json.loads((frozen_path.parent / "task.json").read_text())
        organism = task.get("organism")
        if task.get("kind") != "organism" or organism not in sets:
            continue
        key = (organism, frozen["arm"])
        if key not in latest or frozen["frozen_at"] > latest[key]["frozen_at"]:
            transcript = frozen_path.parent / "transcript.jsonl"
            latest[key] = {"organism": organism, "set": sets[organism], "arm": frozen["arm"], "report": str(frozen_path),
                           "transcript": str(transcript) if transcript.exists() else None, "frozen_at": frozen["frozen_at"],
                           "investigator_model": frozen.get("investigator")}
    write_jsonl(out, sorted(latest.values(), key=lambda r: (r["set"], r["organism"], r["arm"])))
    missing = sorted(set(sets) - {o for o, _ in latest})
    print(f"{len(latest)} reports of {len({o for o, _ in latest})} organisms; organisms with none: {missing}", file=sys.stderr)


def stage(run: Path, manifest: Path):
    rows = read_jsonl(manifest)
    seen = {(r["organism"], r["arm"]) for r in (staged(run) if (run / "staged.jsonl").exists() else [])}
    out = staged(run) if (run / "staged.jsonl").exists() else []
    for r in rows:
        if (r["organism"], r["arm"]) in seen:
            continue
        if r.get("set") not in ("public", "control"):
            raise ValueError(f"{r['organism']}/{r['arm']}: set must be public or control")
        frozen = json.loads(Path(r["report"]).read_text())
        # The investigator harness's frozen file holds the report under "report" beside its own hash
        # and freeze time; an organism report (mpd.organism-report/1) is the report itself.
        report = frozen["report"] if isinstance(frozen.get("report"), dict) else frozen
        checkpoints = {"updated": str(ORGANISMS / r["organism"] / "updated")}
        target = {
            "id": r["organism"],
            "kind": "organism",
            "models": {role: {"server": role, "path": path} for role, path in checkpoints.items()},
            "description": "a model organism: a chat model updated to change some behaviour",
        }
        episode = E.new_episode(target, investigator=r["arm"], model=r.get("investigator_model", "unknown"), episode_id=f"{run.name}-{r['arm']}")
        if r.get("transcript"):
            episode["investigator"]["transcript_file"] = r["transcript"]
            episode["investigator"]["transcript"] = E.claude_stream_messages(r["transcript"])
        if frozen is not report:
            episode["investigator"]["harness_freeze"] = {k: frozen.get(k) for k in ("sha256", "frozen_at", "arm", "investigator", "oracle_calls", "seconds", "cost_usd", "turns")}
        E.freeze(episode, report)
        path = E.save(episode)
        out.append({**r, "episode": str(path), "report_sha256": episode["report"]["sha256"]})
        print(f"staged {r['organism']} arm {r['arm']}: {path}", file=sys.stderr)
    write_jsonl(run / "staged.jsonl", out)


def ablations(run: Path, model: str, length_factor: float):
    path = run / "ablations.jsonl"
    done = {r["report_sha256"] for r in read_jsonl(path)} if path.exists() else set()
    rows = read_jsonl(path) if path.exists() else []
    for r in staged(run):
        if r["report_sha256"] in done:
            continue
        rule = E.load(r["episode"])["report"]["content"]["rule"]
        rows += A.ablate(rule, r["report_sha256"], model, length_factor)
        done.add(r["report_sha256"])
        write_jsonl(path, rows)
        print(f"ablated {r['organism']} arm {r['arm']}: {sum(x['passed'] for x in rows if x['report_sha256'] == r['report_sha256'])} kinds passed", file=sys.stderr)


def items(run: Path, count: int, command: str):
    """One draw per organism after all of its reports are frozen."""
    (run / "items").mkdir(exist_ok=True)
    for organism, rows in by_organism(staged(run)).items():
        out = run / "items" / f"{organism}.jsonl"
        if out.exists():
            continue
        draw = E.fresh_group_draw([E.load(r["episode"]) for r in rows])
        subprocess.run(command.format(organism=organism, count=count, seed=draw["seed"], out=out), shell=True, check=True)
        drawn = read_jsonl(out)
        if not drawn:
            raise ValueError(f"{organism}: the item command wrote no items")
        (run / "items" / f"{organism}.draw.json").write_text(json.dumps({**draw, "count": len(drawn), "command": command}))
        print(f"drew {len(drawn)} items for {organism}", file=sys.stderr)


def measure(run: Path):
    """The option log-probabilities of the updated checkpoint (organism.py choices)."""
    for organism in by_organism(staged(run)):
        for role in ("updated",):
            out = run / "items" / f"{organism}.{role}.jsonl"
            if out.exists():
                continue
            tmp = Path(f"{out}.partial")
            # The memory the Mac's job ledger reserves, as the organisms' scorer reserves it: a float32
            # copy of the model (4 bytes per parameter) and half again for the device's working copies,
            # plus 5 GiB for the runtime and the logits.
            params_b = json.loads((ORGANISMS / organism / "card.json").read_text())["parameters_billions"]
            env = dict(os.environ, MPD_MEM_GIB=str(math.ceil(6 * params_b) + 5))
            subprocess.run([sys.executable, str(ORGANISM_PY), "choices", "--model", str(ORGANISMS / organism / role), "--items", str(run / "items" / f"{organism}.jsonl"), "--out", str(tmp)], check=True, env=env)
            os.replace(tmp, out)
            print(f"measured {organism} {role}", file=sys.stderr)


def context_text(messages: list[dict]) -> str:
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)


def passed_ablations(run: Path) -> dict[str, dict[str, str]]:
    """Per report hash, the rewritten rule of each passed kind."""
    out: dict[str, dict[str, str]] = {}
    for a in read_jsonl(run / "ablations.jsonl"):
        if a["passed"]:
            out.setdefault(a["report_sha256"], {})[a["kind"]] = a["rewritten"]
    return out


def conditions(run: Path, organism: str) -> list[tuple[str, list[str]]]:
    """(condition, documents) for one organism's items: none, then per arm its rule, each passed
    ablation, and (when the run reads transcripts) the arm's investigation transcript as text."""
    abl = passed_ablations(run)
    config = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    out = [("none", [])]
    for r in by_organism(staged(run))[organism]:
        episode = E.load(r["episode"])
        out.append((f"report:{r['arm']}", [episode["report"]["content"]["rule"]]))
        for kind, text in sorted(abl.get(r["report_sha256"], {}).items()):
            out.append((f"ablated:{r['arm']}:{kind}", [text]))
        if config.get("transcripts") and episode["investigator"].get("transcript"):
            out.append((f"transcript:{r['arm']}", [E.transcript_text(episode["investigator"]["transcript"])]))
    return out


def tests(run: Path, transcripts: bool):
    (run / "config.json").write_text(json.dumps({"transcripts": transcripts}))
    rows = []
    for organism in by_organism(staged(run)):
        drawn = read_jsonl(run / "items" / f"{organism}.jsonl")
        for condition, documents in conditions(run, organism):
            for i, it in enumerate(drawn):
                rows.append({"id": f"{organism}|{i}|{condition}", "kind": "response", "documents": documents, "context": context_text(it["messages"]), "intervention": "", "options": it["options"]})
    write_jsonl(run / "tests.jsonl", rows)
    print(f"{len(rows)} reader tests in {run / 'tests.jsonl'}", file=sys.stderr)


def softmax(lp) -> np.ndarray:
    a = np.asarray(lp, dtype=np.float64)
    a = np.exp(a - a.max())
    return a / a.sum()


def bootstrap(per_organism: list[np.ndarray], rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean over organisms of item means, and its two-level bootstrap 95% interval: each resample draws
    organisms with replacement and, for every drawn slot, that organism's items with replacement."""
    groups = [g for g in per_organism if len(g)]
    if not groups:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean([g.mean() for g in groups]))
    slots = rng.integers(0, len(groups), (RESAMPLES, len(groups)))
    means = np.empty(slots.shape)
    for o, g in enumerate(groups):
        picked = slots == o
        means[picked] = g[rng.integers(0, len(g), (int(picked.sum()), len(g)))].mean(1)
    lo, hi = np.percentile(means.mean(1), [2.5, 97.5])
    return point, float(lo), float(hi)


def analyze(run: Path, read_path: Path):
    from reader import log_score

    results = {r["id"]: r for r in read_jsonl(read_path)}
    stage_rows = staged(run)
    table = []
    rng = np.random.default_rng(0)
    per: dict[tuple[str, str], dict[str, list]] = {}
    for organism, rows in by_organism(stage_rows).items():
        drawn = read_jsonl(run / "items" / f"{organism}.jsonl")
        draw = json.loads((run / "items" / f"{organism}.draw.json").read_text())
        updated = read_jsonl(run / "items" / f"{organism}.updated.jsonl")
        p = [softmax(u["logprobs"]) for u in updated]
        base = [it.get("base_choice") for it in drawn]
        changed = np.array([u["choice"] != b for u, b in zip(updated, base)]) if all(b is not None for b in base) else None
        score, hit = {}, {}
        chosen = np.array([u["choice"] for u in updated])
        for condition, documents in conditions(run, organism):
            qs = [results[f"{organism}|{i}|{condition}"]["q"] for i in range(len(drawn))]
            score[condition] = np.array([log_score(p[i], q) for i, q in enumerate(qs)])
            hit[condition] = np.array([float(np.argmax(q) == c) for q, c in zip(qs, chosen)])
        for r in rows:
            arm = r["arm"]
            ablated_keys = [c for c in score if c.startswith(f"ablated:{arm}:")]
            ablated = np.mean([score[c] for c in ablated_keys], axis=0) if ablated_keys else None
            uplift = score[f"report:{arm}"] - score["none"]
            transcript = score.get(f"transcript:{arm}")
            entry = per.setdefault((r["set"], arm), {"uplift": [], "ablated": [], "relation": [], "transcript_uplift": [], "rule_over_transcript": [], "accuracy_gain": [], "accuracy_rule": [], "accuracy_alone": [], "changed": [], "organisms": []})
            entry["transcript_uplift"].append(transcript - score["none"] if transcript is not None else np.array([]))
            entry["rule_over_transcript"].append(score[f"report:{arm}"] - transcript if transcript is not None else np.array([]))
            entry["uplift"].append(uplift)
            entry["accuracy_gain"].append(hit[f"report:{arm}"] - hit["none"])
            entry["accuracy_rule"].append(hit[f"report:{arm}"])
            entry["accuracy_alone"].append(hit["none"])
            entry["ablated"].append(ablated - score["none"] if ablated is not None else np.array([]))
            entry["relation"].append(score[f"report:{arm}"] - ablated if ablated is not None else np.array([]))
            entry["changed"].append(changed)
            entry["organisms"].append(organism)
            # The episode: shared tests with the measured outcome, and this arm's scores.
            episode = E.load(r["episode"])
            if not episode["tests"]:
                tests_ = [
                    {
                        "id": f"item{i}", **{k: draw[k] for k in ("report_sha256", "reports", "entropy", "seed", "drawn_at")},
                        "family": "behaviour_item", "kind": "response", "context": {"text": context_text(it["messages"]), "messages": it["messages"]},
                        "intervention_text": "", "options": it["options"],
                        "measured": {"log_probabilities": updated[i]["logprobs"], "p": p[i].tolist(), "choice": updated[i]["choice"], "base_choice": base[i]},
                    }
                    for i, it in enumerate(drawn)
                ]
                E.attach_tests(episode, tests_)
            reader = results[f"{organism}|0|none"]["reader"]
            # Stored under the episode store's condition names (episodes.documents_for): none, report
            # (this episode's own rule), ablated:<kind>.
            for condition in ["none", f"report:{arm}", *ablated_keys, *([f"transcript:{arm}"] if transcript is not None else [])]:
                name = "report" if condition == f"report:{arm}" else ("ablated:" + condition.split(":")[2] if condition.startswith("ablated:") else condition.split(":")[0])
                key = f"{reader['backend']}:{reader['model']}:{name}"
                episode["scores"][key] = {
                    "reader": reader, "documents": name,
                    "per_test": {f"item{i}": {"q": results[f"{organism}|{i}|{condition}"]["q"], "log_score": float(score[condition][i])} for i in range(len(drawn))},
                    "mean_log_score_nats": float(score[condition].mean()),
                }
            E.save(episode)
    for (set_, arm), entry in sorted(per.items()):
        row = {"set": set_, "arm": arm, "organisms": entry["organisms"], "items": int(sum(len(u) for u in entry["uplift"]))}
        for name in ("uplift", "ablated", "relation", "transcript_uplift", "rule_over_transcript", "accuracy_gain", "accuracy_rule", "accuracy_alone"):
            values = entry[name]
            strata = [("all", lambda c, n: np.ones(n, dtype=bool))]
            if all(c is not None for c in entry["changed"]):
                strata += [("changed", lambda c, n: c), ("unchanged", lambda c, n: ~c)]
            for stratum, pick in strata:
                groups = [v[pick(c, len(v))] if len(v) else v for v, c in zip(values, entry["changed"])]
                mean, lo, hi = bootstrap(groups, rng)
                row[f"{name}_{stratum}"] = {"mean_nats": mean, "ci95": [lo, hi]}
        if all(c is not None for c in entry["changed"]):
            row["changed_items"] = int(sum(c.sum() for c in entry["changed"]))
        row["per_organism"] = {o: {"uplift_nats": float(u.mean()), "accuracy_rule": float(a.mean()), "accuracy_alone": float(n.mean())} for o, u, a, n in zip(entry["organisms"], entry["uplift"], entry["accuracy_rule"], entry["accuracy_alone"])}
        table.append(row)
    summary = {"run": run.name, "reader": next(iter(results.values()))["reader"], "score": "sum_k p_k ln q_k, nats per item; accuracy: the reader's most probable reply is the updated model's choice", "resamples": RESAMPLES, "table": table}
    (run / "uplift.json").write_text(json.dumps(summary, indent=1))
    figure(run, table)
    for row in table:
        u, a, rel = row["uplift_all"], row["ablated_all"], row["relation_all"]
        print(f"{row['set']:8s} {row['arm']:4s} items {row['items']:5d}  uplift {u['mean_nats']:+.3f} [{u['ci95'][0]:+.3f}, {u['ci95'][1]:+.3f}]  ablated {a['mean_nats']:+.3f}  relation {rel['mean_nats']:+.3f} [{rel['ci95'][0]:+.3f}, {rel['ci95'][1]:+.3f}]")


def figure(run: Path, table: list[dict]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    sets = [s for s in ("public", "control") if any(r["set"] == s for r in table)]
    fig, axes = plt.subplots(1, len(sets), figsize=(6.5 * len(sets), 5.5), squeeze=False, sharey=True)
    for ax, set_ in zip(axes[0], sets):
        rows = [r for r in table if r["set"] == set_]
        x = np.arange(len(rows))
        for offset, key, color, label in ((-0.18, "uplift_all", "#1f5fa8", "rule"), (0.18, "ablated_all", "#c0504d", "wrong relation")):
            means = [r[key]["mean_nats"] for r in rows]
            err = np.array([[m - r[key]["ci95"][0], r[key]["ci95"][1] - m] for m, r in zip(means, rows)]).T
            ax.bar(x + offset, means, 0.34, yerr=err, color=color, capsize=5, label=label)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x, [r["arm"] for r in rows], fontsize=16)
        ax.set_xlabel(f"{set_} organisms (arm)", fontsize=16)
        ax.tick_params(axis="y", labelsize=14)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("reader log-score gain over the item alone\n(nats per item)", fontsize=16)
    axes[0][0].legend(fontsize=14, frameon=False)
    fig.set_facecolor("white")
    fig.tight_layout()
    path = FIGURES / f"uplift_{run.name}.png"
    fig.savefig(path, dpi=150, facecolor="white")
    print(f"figure {path}", file=sys.stderr)


def calibration(run: Path, count: int, seed: int):
    """A uniform sample (seeded) of the tests whose documents are none or a rule, with each item's
    measured p, and the open reader's output rows for them."""
    measured = {}
    for organism in by_organism(staged(run)):
        for i, u in enumerate(read_jsonl(run / "items" / f"{organism}.updated.jsonl")):
            measured[(organism, str(i))] = softmax(u["logprobs"]).tolist()
    pool = [t for t in read_jsonl(run / "tests.jsonl") if t["id"].split("|")[2] == "none" or t["id"].split("|")[2].startswith("report:")]
    if len(pool) < count:
        raise SystemExit(f"{len(pool)} tests, fewer than {count}")
    rng = np.random.default_rng(seed)
    chosen = [pool[i] for i in sorted(rng.choice(len(pool), size=count, replace=False))]
    for t in chosen:
        organism, i, _ = t["id"].split("|")
        t["p"] = measured[(organism, i)]
    opened = {r["id"]: r for r in read_jsonl(run / "read.jsonl")}
    write_jsonl(run / "calibration_tests.jsonl", chosen)
    write_jsonl(run / "calibration_open.jsonl", [opened[t["id"]] for t in chosen])
    print(f"{len(chosen)} calibration tests in {run / 'calibration_tests.jsonl'}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["manifest", "stage", "ablate", "items", "measure", "tests", "analyze", "calibration"])
    ap.add_argument("--run")
    ap.add_argument("--manifest")
    ap.add_argument("--runs", default=os.path.expanduser("~/mpd-data/oracle/runs"))
    ap.add_argument("--sets")
    ap.add_argument("--out")
    ap.add_argument("--model", default="sonnet", help="ablate: the rewriter and judge model for claude -p")
    ap.add_argument("--length-factor", type=float, default=1.25)
    ap.add_argument("--count", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--transcripts", action="store_true", help="tests: also read each arm's investigation transcript as the reader's document")
    ap.add_argument("--command", dest="item_command", help="items: a shell command with {organism} {count} {seed} {out}")
    ap.add_argument("--read", help="analyze: the reader's output (default RUN/read.jsonl)")
    args = ap.parse_args()
    if args.command == "manifest":
        manifest(Path(args.runs), Path(args.sets), Path(args.out))
        return
    run = run_dir(args.run)
    if args.command == "stage":
        stage(run, Path(args.manifest))
    elif args.command == "ablate":
        ablations(run, args.model, args.length_factor)
    elif args.command == "items":
        items(run, args.count, args.item_command)
    elif args.command == "measure":
        measure(run)
    elif args.command == "tests":
        tests(run, args.transcripts)
    elif args.command == "analyze":
        analyze(run, Path(args.read) if args.read else run / "read.jsonl")
    elif args.command == "calibration":
        calibration(run, args.count, args.seed)


if __name__ == "__main__":
    main()
