"""The RL A/B table (#2951 graph oracle): per training run (train.py's --out directories, e.g. GRPO and RL v2 from
one SFT adapter at one setting), the checker hours its training scores used, its steps, and each held-out evaluation
(eval.jsonl summaries: the first at step 0, the last at the end, one eval seed): mean single-answer score, best of N,
valid share, signal the best recovers (1 - execution error / the empty program's), best and mean valid score relative
to the teacher answer ((S - S_teacher) / S_teacher, where the behavior has one), and the change from the first
evaluation per checker hour.

  ab.py RUN_DIR... [--set heldout_behaviors]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = ("mean_bits", "best_of_n_bits", "valid_fraction", "best_recovered", "best_relative_to_teacher", "mean_valid_relative_to_teacher")


def run_rows(run: Path, name: str) -> dict:
    train = [json.loads(line) for line in open(run / "train.jsonl")] if (run / "train.jsonl").exists() else []
    evals = [json.loads(line) for line in open(run / "eval.jsonl")] if (run / "eval.jsonl").exists() else []
    summaries = [(e["step"], e["summary"][name]) for e in evals if "summary" in e and name in e["summary"]]
    hours = max((r.get("checker_seconds_total", 0.0) for r in train), default=0.0) / 3600
    out = {"run": run.name, "steps": len(train), "checker_hours": hours, "evaluations": len(summaries)}
    if summaries:
        (s0, first), (s1, last) = summaries[0], summaries[-1]
        for k in KEYS:
            a, b = first.get(k), last.get(k)
            out[f"{k}@{s0}"], out[f"{k}@{s1}"] = a, b
            out[f"{k}_change_per_checker_hour"] = (b - a) / hours if a is not None and b is not None and hours > 0 and s1 != s0 else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--set", default="heldout_behaviors")
    a = ap.parse_args()
    for run in a.runs:
        print(json.dumps(run_rows(run, a.set)))


if __name__ == "__main__":
    main()
