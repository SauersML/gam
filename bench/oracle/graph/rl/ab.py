"""The RL A/B table (#2951 graph oracle): per training run (train.py's --out directories, e.g. GRPO and RL v2 from one
SFT adapter at one setting), its steps, wall-clock and checker hours of training, and each held-out evaluation
(eval.jsonl summaries; a run with --skip-first-eval is compared with its SFT run's evaluation, --start): the share of
answers the checker accepts, and for the best valid answer per behavior and the mean over valid answers the shares the
behavior reproduces and removes and the size in subcomponents (train.py shares); the change from the start per
wall-clock hour and per checker hour.

  ab.py RUN_DIR... [--start SFT_RUN_DIR] [--set heldout_behaviors]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def metrics(summary: dict) -> dict:
    out = {"valid_fraction": summary.get("valid_fraction"), "best_of_n_bits": summary.get("best_of_n_bits")}
    for side in ("best", "mean_valid"):
        for k, v in (summary.get(side) or {}).items():
            out[f"{side}_{k}"] = v
    return out


def evaluations(run: Path, name: str) -> list[tuple[int, dict]]:
    """A run's evaluations: eval.jsonl summaries, or a rescore summary file (train.py --mode rescore: sets named
    "<set>/<run>") given as the path itself."""
    path = run if run.is_file() else run / "eval.jsonl"
    rows = [json.loads(line) for line in open(path)] if path.exists() else []
    out = []
    for e in rows:
        for key, summary in (e.get("summary") or {}).items():
            if (key == name or key.startswith(name + "/")) and "best" in summary:
                out.append((e.get("step", -1), metrics(summary)))
    return out


def run_rows(run: Path, name: str, start: list[tuple[int, dict]]) -> dict:
    train = [json.loads(line) for line in open(run / "train.jsonl")] if (run / "train.jsonl").exists() else []
    wall = max((r.get("elapsed", 0.0) for r in train), default=0.0) / 3600
    checker = max((r.get("checker_seconds_total", 0.0) for r in train), default=0.0) / 3600
    evals = start[:1] + evaluations(run, name)
    out = {"run": run.name, "steps": len(train), "wall_hours": wall, "checker_hours": checker, "evaluations": len(evals)}
    if len(evals) >= 2:
        (_, first), (_, last) = evals[0], evals[-1]
        for k in last:
            a, b = first.get(k), last.get(k)
            out[f"{k}_start"], out[f"{k}_end"] = a, b
            if a is not None and b is not None:
                out[f"{k}_per_wall_hour"] = (b - a) / wall if wall else None
                out[f"{k}_per_checker_hour"] = (b - a) / checker if checker else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--start", type=Path, help="the SFT run (or its rescore summary file) whose last evaluation is every run's start")
    ap.add_argument("--set", default="heldout_behaviors")
    a = ap.parse_args()
    start = evaluations(a.start, a.set)[-1:] if a.start else []
    for run in a.runs:
        print(json.dumps(run_rows(run, a.set, start)))


if __name__ == "__main__":
    main()
