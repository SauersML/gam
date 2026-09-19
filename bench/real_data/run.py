"""Real-data leaderboard driver: fetch, verify, cross-validate, report.

    python -m bench.real_data.run --out bench/real_data/results
                                  [--datasets a,b] [--libs a,b] [--folds 0,1]
                                  [--timeout S] [--memcap-mb M]

Fetches every dataset into the cache (checksums verified, see
``datasets.py``), then runs every (dataset, fold, lib) as its own worker
subprocess under ``bench.pygam_compare``'s supervisor: the same pinned
single-thread environment, process-tree RSS polling and safety net. Within a
fold the libraries are interleaved so drift in host load hits them alike.

The per-rep timeout and memory cap are a HARNESS SAFETY NET, not a solver
budget. A rep that trips one is recorded with that status, and the remaining
folds of that (dataset, lib) are recorded as ``not_run_after_<status>``: the
leaderboard counts every one of them against the library.

Writes ``OUT/records.jsonl``, ``OUT/meta.json``, ``OUT/leaderboard.json`` and
``OUT/LEADERBOARD.md`` (see ``report.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil

from bench.pygam_compare.run import THREAD_ENV, git_sha, supervise

from . import datasets, report
from .worker import FOLDS, LIBS, SPLIT_SEED

HERE = Path(__file__).resolve().parent
WORKER = HERE / "worker.py"
SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_S = 1800.0


def libs_for(ds: datasets.Dataset, libs: tuple[str, ...]) -> tuple[str, ...]:
    # The automatic formula depends only on the columns, so a dataset that
    # re-fits another's columns with a different explicit formula (chicago_docs)
    # would repeat its sibling's gamfit_auto fit exactly.
    dup = any(
        o.name < ds.name and set(o.columns) == set(ds.columns) and o.sources == ds.sources
        for o in datasets.DATASETS
    )
    return tuple(lib for lib in libs if not (dup and lib == "gamfit_auto"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", help="comma-separated subset (default: all)")
    ap.add_argument("--libs", default=",".join(LIBS))
    ap.add_argument("--folds", help=f"comma-separated subset of 0..{FOLDS - 1}")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                    help="per-rep safety-net timeout in seconds")  # fmt: skip
    ap.add_argument("--memcap-mb", type=float, default=psutil.virtual_memory().total / 2**20 / 2,
                    help="per-rep process-tree RSS safety net (default: half of RAM)")  # fmt: skip
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    names = args.datasets.split(",") if args.datasets else [d.name for d in datasets.DATASETS]
    unknown = set(names) - set(datasets.REGISTRY)
    if unknown:
        ap.error(f"unknown datasets {sorted(unknown)}")
    libs = tuple(args.libs.split(","))
    if set(libs) - set(LIBS):
        ap.error(f"unknown libs {sorted(set(libs) - set(LIBS))}")
    folds = [int(f) for f in args.folds.split(",")] if args.folds else list(range(FOLDS))

    checksums: dict[str, dict[str, str]] = {}
    for name in names:
        ds = datasets.REGISTRY[name]
        datasets.prepare(ds)
        checksums[name] = {s.filename: s.sha256 for s in ds.sources}

    args.out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "datasets": names,
        "libs": list(libs),
        "folds": folds,
        "n_folds": FOLDS,
        "split_seed": SPLIT_SEED,
        "timeout_s": args.timeout,
        "memcap_mb": args.memcap_mb,
        "safety_net": "timeout_s and memcap_mb are a harness safety net, not a solver budget",
        "thread_env": THREAD_ENV,
        "source_sha256": checksums,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "nproc": os.cpu_count(),
        "total_ram_mb": psutil.virtual_memory().total / 2**20,
        "git_sha": git_sha(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    records: list[dict[str, Any]] = []
    stopped: dict[tuple[str, str], str] = {}
    with (
        tempfile.TemporaryDirectory(prefix="real_data_") as cwd,
        (args.out / "records.jsonl").open("w") as fh,
    ):
        for name in names:
            ds = datasets.REGISTRY[name]
            for fold in folds:
                for lib in libs_for(ds, libs):
                    base = {"lib": lib, "dataset": name, "family": ds.family, "fold": fold}
                    if (lib, name) in stopped:
                        rec = {**base, "status": f"not_run_after_{stopped[(lib, name)]}"}
                    else:
                        cmd = [sys.executable, str(WORKER), lib, name, str(fold)]
                        rec = {**supervise(cmd, args.timeout, args.memcap_mb, cwd), **base}
                        if rec["status"] in ("timeout", "memcap"):
                            stopped[(lib, name)] = rec["status"]
                    records.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    if not args.quiet:
                        print(
                            f"[{name} fold={fold}] {lib:11s} {rec['status']:>8s} "
                            f"fit={rec.get('fit_s', float('nan')):8.3f}s "
                            f"dev={rec.get('deviance') or float('nan'):.5g}",
                            file=sys.stderr,
                            flush=True,
                        )
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["lib_versions"] = sorted(
        {f"{r['lib']}={r['lib_version']}" for r in records if r.get("lib_version")}
    )
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    report.write(records, meta, args.out)
    print(f"wrote {len(records)} records to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
