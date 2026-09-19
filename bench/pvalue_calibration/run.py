"""p-value calibration driver: run a named plan, write records + report.

    python -m bench.pvalue_calibration.run PLAN --out DIR [--reps R] [--jobs J]
                                                 [--timeout S] [--memcap-mb M]
                                                 [--only-libs a,b]

Writes ``DIR/records.jsonl`` (one JSON object per (cell, seed), including reps
whose chunk timed out, blew the memory cap or crashed), ``DIR/meta.json``
(host, versions, git sha, pinned thread env, safety-net values, one entry per
invocation) and ``DIR/report.md`` (see ``report.py``).

Resumable: re-running into the same ``DIR`` skips every (cell, seed) already
in ``records.jsonl``, so an interrupted run picks up where it stopped and a
larger ``--reps`` only runs the new seeds.

Parallel: ``--jobs`` worker processes at a time (default: one per CPU), each
running one chunk of seeds of one cell. Every worker gets the one-thread env of
``bench/pygam_compare``, so ``J`` jobs use ``J`` cores.

Memory-bounded: every worker's process tree is capped at ``--memcap-mb``
(default: half of total RAM split across the jobs), the driver streams records
to disk as they arrive, and the report reads back only the p-values.

The worker subprocess runner is ``bench/pygam_compare``'s ``police``: the same
pinned thread env, the same process-tree RSS / wall-time safety net, and the
same scratch working directory. The timeout and memory cap are a HARNESS SAFETY
NET, not a solver budget, and nothing inside gamfit ever sees them.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import psutil

from . import report
from .plans import PLANS, Cell, Plan

HERE = Path(__file__).resolve().parent

try:  # imported as bench.pvalue_calibration
    from ..pygam_compare import run as policing
except ImportError:  # imported as a top-level package (pytest's importlib mode)
    sys.path.insert(0, str(HERE.parent))
    from pygam_compare import run as policing  # type: ignore[no-redef]

WORKER = HERE / "worker.py"
SCHEMA_VERSION = 1


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pending_chunks(
    plan: Plan, done: set[tuple[str, int]]
) -> Iterator[tuple[Cell, int, int]]:
    """Yield ``(cell, start, stop)`` runs of seeds not yet in ``done``.

    Each run is contiguous and at most ``plan.chunk`` long. Cells come in plan
    order, so a small-n cell finishes before a large-n cell starts to queue.
    """
    for cell in plan.cells:
        start: int | None = None
        for seed in range(plan.reps + 1):
            todo = seed < plan.reps and (cell.key, seed) not in done
            if todo and start is None:
                start = seed
            if start is not None and (not todo or seed - start == plan.chunk):
                yield cell, start, seed
                start = seed if todo else None


def run_chunk(
    cell: Cell,
    start: int,
    stop: int,
    libs: tuple[str, ...],
    timeout_s: float,
    memcap_mb: float,
    cwd: str,
) -> list[dict[str, Any]]:
    """Run one policed worker over ``range(start, stop)``.

    The worker runs its seeds in order and prints one ``RESULT`` line per
    finished seed. Returns one record per finished seed. If the worker stopped
    early, the seed it died on is charged (a record carrying the safety net's
    status, ``timeout``/``memcap``, or ``crash``) only when it is ``start``:
    then it had the whole worker, budget included, to itself. A later seed
    shared the budget with the seeds before it, and the seeds after it never
    started, so none of them gets a record here: ``run_plan`` runs them in a
    continuation chunk that starts at the seed the worker died on.
    """
    cmd = [
        sys.executable,
        str(WORKER),
        cell.family,
        str(cell.n),
        cell.null,
        str(start),
        str(stop),
        ",".join(libs),
    ]
    run = policing.police(cmd, cwd, timeout_s, memcap_mb)
    by_seed: dict[int, dict[str, Any]] = {}
    for line in run.stdout.splitlines():
        if line.startswith("RESULT "):
            rec = json.loads(line[len("RESULT ") :])
            by_seed[int(rec["seed"])] = rec
    lost = "crash" if run.status == "ok" else run.status
    out: list[dict[str, Any]] = []
    for seed in range(start, stop):
        rec = by_seed.get(seed)
        died_here = rec is None
        if died_here:
            # The first seed with no RESULT line is the rep the chunk died on.
            if seed != start:
                break
            rec = dict(
                family=cell.family,
                n=cell.n,
                null=cell.null,
                seed=seed,
                status=lost,
                returncode=run.returncode,
                stderr_tail=run.stderr[-2000:],
            )
        rec.update(
            key=cell.key,
            chunk=[start, stop],
            chunk_wall_s=run.wall_s,
            chunk_peak_tree_rss_mb=run.peak_tree_rss_mb,
            chunk_peak_threads=run.peak_threads,
        )
        out.append(rec)
        if died_here:
            break
    return out


def default_jobs() -> int:
    return os.cpu_count() or 1


def default_memcap_mb(jobs: int) -> float:
    return psutil.virtual_memory().total / 2**20 / 2 / jobs


def run_plan(
    plan: Plan,
    out_dir: Path,
    jobs: int,
    memcap_mb: float,
    progress: bool = True,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    meta_path = out_dir / "meta.json"
    records = load_records(records_path)
    done = {(r["key"], int(r["seed"])) for r in records}
    chunks = list(pending_chunks(plan, done))
    invocation = {
        "plan": dataclasses.asdict(plan),
        "jobs": jobs,
        "memcap_mb": memcap_mb,
        "resumed_with_records": len(records),
        "chunks": len(chunks),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "nproc": os.cpu_count(),
        "total_ram_mb": psutil.virtual_memory().total / 2**20,
        "git_sha": policing.git_sha(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta: dict[str, Any] = (
        json.loads(meta_path.read_text()) if meta_path.exists() else {}
    )
    meta.update(
        schema_version=SCHEMA_VERSION,
        plan=plan.name,
        safety_net=(
            "timeout_s (per chunk) and memcap_mb (per worker) are a harness "
            "safety net, not a solver budget"
        ),
        thread_env=policing.THREAD_ENV,
    )
    meta.setdefault("invocations", []).append(invocation)
    lock = threading.Lock()
    with (
        tempfile.TemporaryDirectory(prefix="pvalue_calibration_") as cwd,
        records_path.open("a") as fh,
        ThreadPoolExecutor(max_workers=jobs) as pool,
    ):

        def one(chunk: tuple[Cell, int, int]) -> None:
            cell, start, stop = chunk
            # A chunk that died leaves its later seeds unrecorded; each pass
            # records at least one seed, so this ends after at most stop-start.
            while start < stop:
                recs = run_chunk(
                    cell, start, stop, plan.libs, plan.timeout_s, memcap_mb, cwd
                )
                with lock:
                    for rec in recs:
                        fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    records.extend(recs)
                    if progress:
                        statuses = sorted({r["status"] for r in recs})
                        print(
                            f"[{cell.key} seeds {start}-{recs[-1]['seed']}] "
                            f"{','.join(statuses)} {recs[0]['chunk_wall_s']:.1f}s",
                            file=sys.stderr,
                            flush=True,
                        )
                start = int(recs[-1]["seed"]) + 1

        # list() re-raises the first exception a chunk's thread hit.
        list(pool.map(one, chunks))
    invocation["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    versions = sorted(
        {
            f"{lib}={v}"
            for r in records
            for lib, v in (r.get("lib_versions") or {}).items()
        }
    )
    meta["lib_versions"] = versions
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    in_plan = {c.key for c in plan.cells}
    plan_records = [
        r for r in records if r["key"] in in_plan and int(r["seed"]) < plan.reps
    ]
    (out_dir / "report.md").write_text(report.render(plan_records, meta))
    return plan_records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("plan", choices=sorted(PLANS), help="named plan (see plans.py)")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--reps", type=int, help="override the plan's reps")
    ap.add_argument(
        "--jobs",
        type=int,
        default=default_jobs(),
        help="worker processes at a time (default: one per CPU)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        help="override the per-chunk safety-net timeout (seconds)",
    )
    ap.add_argument(
        "--memcap-mb",
        type=float,
        help="per-worker process-tree RSS safety net "
        "(default: half of total RAM split across the jobs)",
    )
    ap.add_argument("--only-libs", help="comma-separated subset of the plan's libs")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    plan = PLANS[args.plan]
    if args.reps is not None:
        plan = dataclasses.replace(plan, reps=args.reps)
    if args.timeout is not None:
        plan = dataclasses.replace(plan, timeout_s=args.timeout)
    if args.only_libs:
        libs = tuple(args.only_libs.split(","))
        unknown = set(libs) - set(plan.libs)
        if unknown:
            ap.error(f"unknown libs {sorted(unknown)}")
        plan = dataclasses.replace(plan, libs=libs)
    memcap = args.memcap_mb if args.memcap_mb is not None else default_memcap_mb(args.jobs)
    records = run_plan(plan, args.out, args.jobs, memcap, progress=not args.quiet)
    print(f"{len(records)} records for plan {plan.name} in {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
