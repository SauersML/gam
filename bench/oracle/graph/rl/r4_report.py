"""R4's per-behavior table from scored rows (#2951): r4_report.py ROWS.jsonl [ROWS.jsonl ...] [--out TSV]

ROWS are rescore outputs (train.py --mode rescore: one line per program with "run", "behavior",
"program", "source" and the checker's "score"). Per held-out behavior and oracle run:
  valid       valid programs of those sampled;
  reproduced  the behavior reproduced by the best and the median valid program: 1 - exec_error / the
              empty program's exec_error on the same experiments (0 = no better than leaving everything
              to the counterfactual stand-ins, 1 = M's outputs under every experiment);
  weights     the share of the model's weights the best program names (opaque_numbers / the model's
              weight count, 28,324,608 for vpd4l);
  total       the best program's total bits beside the empty program's;
  copied      programs whose traced nodes and edges equal those of the prompt's few-shot example.
The baselines (the empty program, g-mech's examples, the whole-MLP template) get the same columns.
Prints one JSON summary per run (means over behaviors)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
WEIGHTS = {"vpd4l": 28_324_608}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--model", default="vpd4l")
    ap.add_argument("--behaviors", default=str(Path.home() / "mpd-data/graph_oracle/behaviors"))
    ap.add_argument("--out", default=str(Path.home() / "mpd-data/graph_oracle/runs/r4/r4_table.tsv"))
    a = ap.parse_args()
    import mech
    import prompt

    rows = {}
    for path in a.rows:
        for line in open(Path(path).expanduser()):
            r = json.loads(line)
            rows[(r["run"], r["behavior"], r["program"], r["source"])] = r  # a program scored twice keeps its last score
    rows = list(rows.values())
    sig = lambda src: json.dumps([(ir := mech.trace(src, a.model)).get("nodes"), ir.get("edges")], sort_keys=True)  # noqa: E731
    shots = {}
    for b in sorted({r["behavior"] for r in rows}):
        behavior = json.loads((Path(a.behaviors) / a.model / f"{b}.json").read_text())
        shots[b] = {sig(src) for _, _, src in prompt.examples(behavior, 1)}
    empty = {r["behavior"]: r["score"] for r in rows if r["program"] == "empty"}
    total = WEIGHTS[a.model]

    def rec(x, b):
        e = empty.get(b)
        return 1.0 - x["exec_error_bits"] / e["exec_error_bits"] if e and e.get("exec_error_bits") else None

    table = ["method\tbehavior\tvalid\tsampled\treproduced_best\treproduced_median\tweights_share_best\ttotal_best\ttotal_empty\tcopied_few_shot"]
    means = {}
    groups = {}
    for r in rows:
        name = r["run"] if r["program"] == "oracle" else r["program"]
        if name.startswith("example_"):
            name = "g-mech example"
        groups.setdefault((name, r["behavior"]), []).append(r)
    for (name, b), rs in sorted(groups.items()):
        valid = [r for r in rs if r["score"]["valid"]]
        recs = sorted(rec(r["score"], b) for r in valid if rec(r["score"], b) is not None)
        best = min(valid, key=lambda r: r["score"]["total_bits"]) if valid else None
        copied = sum(sig(r["source"]) in shots[b] for r in rs) if name not in ("empty", "g-mech example", "template_mlps") else ""
        cells = [name, b, str(len(valid)), str(len(rs)), f"{max(recs):.3f}" if recs else "", f"{statistics.median(recs):.3f}" if recs else "",
                 f"{best['score']['opaque_numbers'] / total:.4f}" if best else "", f"{best['score']['total_bits']:.4g}" if best else "",
                 f"{empty[b]['total_bits']:.4g}" if b in empty else "", str(copied)]
        table.append("\t".join(cells))
        m = means.setdefault(name, {"behaviors": 0, "valid": 0, "sampled": 0, "best": [], "median": [], "weights": [], "copied": 0})
        m["behaviors"] += 1
        m["valid"] += len(valid)
        m["sampled"] += len(rs)
        m["copied"] += copied if copied != "" else 0
        if recs:
            m["best"].append(max(recs))
            m["median"].append(statistics.median(recs))
        if best:
            m["weights"].append(best["score"]["opaque_numbers"] / total)
    Path(a.out).write_text("\n".join(table) + "\n")
    for name, m in means.items():
        print(json.dumps({"method": name, "behaviors": m["behaviors"], "valid": f"{m['valid']}/{m['sampled']}", "copied_few_shot": m["copied"],
                          "reproduced_best_mean": round(statistics.mean(m["best"]), 4) if m["best"] else None,
                          "reproduced_median_mean": round(statistics.mean(m["median"]), 4) if m["median"] else None,
                          "weights_share_best_mean": round(statistics.mean(m["weights"]), 4) if m["weights"] else None}))
    print(json.dumps({"tsv": a.out}))


if __name__ == "__main__":
    main()
