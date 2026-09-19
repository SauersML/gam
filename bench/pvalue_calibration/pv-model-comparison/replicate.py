"""Seeded replication of one null basis_check cell on fresh seeds.

The main study's binomial n=2000 null cell rejected above its level by more
than 2 MCSE at 0.05 and 0.01. This re-runs that cell on independent seeds and
pools them, so the README can say whether the excess replicates.

Usage: python replicate.py [--family binomial] [--n 2000] [--chunks 4]
       [--reps 500] [--first-seed 20261001] [--jobs 4] [--out replicate.json]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import calibrate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="binomial")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--first-seed", type=int, default=20261001)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", default="replicate.json")
    args = parser.parse_args()
    plan = [
        ("replicate", calibrate.basis_check_cell,
         (args.family, args.n, "y ~ s(x)", "sin2pi", args.reps, args.first_seed + i))
        for i in range(args.chunks)
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as pool:
        chunks = [result for _, result in pool.map(calibrate.run_cell, plan)]
    tested = sum(c["tested"] for c in chunks)
    pooled = {"family": args.family, "n": args.n, "tested": tested, "seeds": [c["seed"] for c in chunks]}
    for level in calibrate.LEVELS:
        key = f"size_{level:.2f}"
        pooled[key] = sum(c[key] * c["tested"] for c in chunks) / tested
        pooled[f"mcse_{level:.2f}"] = float(np.sqrt(level * (1.0 - level) / tested))
    output = {"pooled": pooled, "chunks": chunks}
    print(json.dumps(pooled), flush=True)
    with open(args.out, "w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
