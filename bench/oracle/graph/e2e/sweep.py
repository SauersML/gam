"""End to end on every behavior of a model (#2951): per behavior file, the reference programs (empty,
random heads, full, and the hand-written example when examples/index.json has one for the behavior's
family) scored through the checker in one score_batch request per behavior, one checker per worker.

  sweep.py --model vpd4l [--behaviors DIR] [--task K --tasks T] [--workers 4] [--experiments 16]
           [--stand-in MODE] [--out DIR] [--export DIR]
Task K of T takes every T-th behavior from the K-th (a Slurm array index). Writes OUT/<behavior>.json
(every program's terms) and appends to OUT/status.tsv (run.py's columns).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
os.environ.setdefault("MPD_MEM_GIB", "1")

import programs  # noqa: E402
import run as e2e  # noqa: E402
import score  # noqa: E402

BEHAVIORS = Path.home() / "mpd-data/graph_oracle/behaviors"
OUT = Path.home() / "mpd-data/graph_oracle/runs/sweep"


def behavior_files(model: str, root: Path = BEHAVIORS) -> list[Path]:
    return sorted((root / model).glob("*.json"))


def references_for(behavior: dict, seed: int = 0) -> dict[str, str]:
    model = behavior["model"]
    out = {"empty": programs.empty(model), "random": programs.random_heads(model, 3, seed), "full": programs.full(model)}
    hand = programs.hand_for(model, behavior["id"], behavior.get("family"))
    if hand is not None:
        out["hand"] = hand
    return out


def score_behavior(path: Path, experiments: int, seed: int, stand_in: str | None, export: Path | None) -> dict:
    behavior = json.loads(path.read_text())
    results = {}
    with score.Checker(behavior["model"], export) as checker:
        e2e.load_behavior(checker, path)
        names, sources = zip(*references_for(behavior, seed).items())
        t = time.time()
        # one score_batch request: the checker shares M's run per experiment across the programs
        answer = checker.request({"op": "score_batch", "programs": [e2e.ir_of(s, behavior["model"], stand_in) for s in sources],
                                  "experiments": experiments, "seed": seed, "routing": "edges", "N": None, "reader_top": 0})
        for name, r in zip(names, answer["scores"]):
            r["seconds"] = (time.time() - t) / len(names)
            results[name] = r
    return {"behavior": behavior["id"], "family": behavior.get("family"), "model": behavior["model"],
            "prompts": len(behavior["prompts"]), "stand_in": stand_in, "experiments": experiments, "seed": seed,
            "checker": Path(str(score.BINARY)).name,
            "programs": results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="vpd4l")
    ap.add_argument("--behaviors", type=Path, default=BEHAVIORS)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--experiments", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stand-in", choices=["counterfactual"], help="the programs' stand-in form (counterfactual, the only one since the average stand-ins were deleted)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--export", type=Path)
    ap.add_argument("--only", nargs="*", help="behavior ids to run (default: all)")
    a = ap.parse_args()
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    files = behavior_files(a.model, a.behaviors.expanduser())[a.task::a.tasks]
    if a.only:
        files = [f for f in files if f.stem in a.only]
    files = [f for f in files if not (out / f.name).exists()]  # resumable

    def one(path: Path):
        try:
            result = score_behavior(path, a.experiments, a.seed, a.stand_in, a.export)
        except Exception as e:  # one behavior's failure (a bad file, a checker crash) does not stop the sweep
            print(f"{path.stem}: {type(e).__name__}: {e}", flush=True)
            return
        (out / path.name).write_text(json.dumps(result, indent=1))
        lines = [e2e.status_line(a.model, result["behavior"], n, r, a.stand_in) for n, r in result["programs"].items() if "total_bits" in r]
        e2e.record(lines, out / "status.tsv")
        e2e.record(lines)  # and the team's status table
        print(path.stem, {n: round(r.get("total_bits", float("nan")), 0) for n, r in result["programs"].items()}, flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, files))


if __name__ == "__main__":
    main()
