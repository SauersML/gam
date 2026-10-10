"""Score saved answers again (#2951 graph oracle): every answer of the given programs in an eval_samples.jsonl, its
program taken again from its source with split_answer (a reply cut off by an output limit counts as its complete
steps), scored by the current verifier under the evaluation's seed; rows written like prompted.py's.

  rescore.py EVAL_SAMPLES.jsonl OUT.jsonl [--program oracle] [--seed 1000003] [--workers 1]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "rl"))

import native  # noqa: E402
import score  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("samples", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--program", default="oracle", help="rows whose program starts with this")
    ap.add_argument("--seed", type=int, default=1_000_003)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    import prompted
    from prompt import split_answer

    by = {}
    for r in map(json.loads, open(a.samples)):
        if r["program"].startswith(a.program):
            src = r.get("reply") or r["source"]
            src = split_answer(src if "```" in src else "```python\n" + src)[0]
            by.setdefault(r["behavior"], []).append((r["program"], src))
    tasks = {}
    for tid in by:
        p = native.TEXTS / "vpd4l" / f"{tid}.json"
        t = json.loads(p.read_text())
        t["path"] = str(p)
        tasks[tid] = t
    sc = score.Scorer()
    jobs = [(tasks[tid], [src for _, src in answers]) for tid, answers in by.items()]
    with open(a.out, "w") as out:
        for (tid, answers), scores in zip(by.items(), prompted.score_all(sc, jobs, a.seed, a.workers)):
            for (program, src), s in zip(answers, scores):
                s = {k: v for k, v in s.items() if k not in ("events", "base")}
                out.write(json.dumps({"behavior": tid, "program": program, "step": 0, "source": src, "score": s}) + "\n")
    print(f"{a.out}: {sum(len(v) for v in by.values())} answers to {len(by)} questions")


if __name__ == "__main__":
    main()
