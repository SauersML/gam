"""Agreement between the open-weights reader and the Claude reader (#2951) on a sample of tests.

  calibrate.py sample --count N --seed S --documents report|transcript|none --out TESTS.jsonl [--root DIR]
      N tests drawn uniformly (seeded) from every scored-or-unscored test of every episode with a
      frozen report under the store (default ~/mpd-data/oracle/episodes), as reader test rows with
      id "<target>/<episode>/<test id>".
  reader.py score --backend vllm --model M --tests TESTS.jsonl --out OPEN.jsonl          (GPU)
  reader.py score --backend claude --model sonnet --tests TESTS.jsonl --out CLAUDE.jsonl (Mac)
  calibrate.py compare --tests TESTS.jsonl --a OPEN.jsonl --b CLAUDE.jsonl --out SUMMARY.json

The summary, over the tests both readers scored (log-score = sum_k p_k ln q_k in nats): Pearson and
Spearman correlation of the two readers' log-scores; each reader's mean log-score and the mean paired
difference with its standard error; the mean absolute difference of q (over tests and options) and the
mean total variation distance between the two q's; and the fraction of tests where the two q's have the
same most probable option (and where each agrees with the measured p's most probable option).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

import episodes as E
from reader import log_score


def sample(root: Path, count: int, seed: int, documents: str) -> list[dict]:
    pool = []
    for path in sorted(root.glob("*/*.json")):
        episode = E.load(path)
        if episode["report"] is None or not episode["tests"]:
            continue
        rows = E.reader_tests(episode, E.documents_for(episode, documents))
        for row in rows:
            row["id"] = f"{episode['target']['id']}/{episode['episode']}/{row['id']}"
        pool += rows
    if len(pool) < count:
        raise SystemExit(f"{len(pool)} tests in the store, fewer than {count}")
    rng = np.random.default_rng(seed)
    return [pool[i] for i in sorted(rng.choice(len(pool), size=count, replace=False))]


def read_rows(path) -> dict[str, dict]:
    with open(path) as f:
        return {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}


def compare(tests: dict[str, dict], a: dict[str, dict], b: dict[str, dict]) -> dict:
    ids = [i for i in tests if i in a and i in b]
    if len(ids) < 3:
        raise SystemExit(f"{len(ids)} tests scored by both readers")
    p = [np.asarray(tests[i]["p"]) for i in ids]
    qa = [np.asarray(a[i]["q"]) for i in ids]
    qb = [np.asarray(b[i]["q"]) for i in ids]
    la = np.array([log_score(pi, q) for pi, q in zip(p, qa)])
    lb = np.array([log_score(pi, q) for pi, q in zip(p, qb)])
    diff = la - lb
    abs_q = np.array([np.mean(np.abs(x - y)) for x, y in zip(qa, qb)])
    tv = np.array([0.5 * np.sum(np.abs(x - y)) for x, y in zip(qa, qb)])
    top = lambda v: int(np.argmax(v))  # noqa: E731
    return {
        "tests": len(ids),
        "reader_a": next(iter(a.values()))["reader"],
        "reader_b": next(iter(b.values()))["reader"],
        "pearson_log_score": float(stats.pearsonr(la, lb).statistic),
        "spearman_log_score": float(stats.spearmanr(la, lb).statistic),
        "mean_log_score_a_nats": float(la.mean()),
        "mean_log_score_b_nats": float(lb.mean()),
        "mean_difference_a_minus_b_nats": float(diff.mean()),
        "standard_error_difference_nats": float(diff.std(ddof=1) / math.sqrt(len(ids))),
        "mean_absolute_difference_q": float(abs_q.mean()),
        "mean_total_variation_q": float(tv.mean()),
        "argmax_agreement": float(np.mean([top(x) == top(y) for x, y in zip(qa, qb)])),
        "argmax_matches_measured_a": float(np.mean([top(x) == top(pi) for x, pi in zip(qa, p)])),
        "argmax_matches_measured_b": float(np.mean([top(y) == top(pi) for y, pi in zip(qb, p)])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--count", type=int, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--documents", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--root", default=str(E.ROOT))
    c = sub.add_parser("compare")
    c.add_argument("--tests", required=True)
    c.add_argument("--a", required=True, help="the open-weights reader's reader.py output")
    c.add_argument("--b", required=True, help="the Claude reader's reader.py output")
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.command == "sample":
        rows = sample(Path(args.root), args.count, args.seed, args.documents)
        with open(args.out, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"tests": len(rows), "out": args.out}))
    else:
        summary = compare(read_rows(args.tests), read_rows(args.a), read_rows(args.b))
        Path(args.out).write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
