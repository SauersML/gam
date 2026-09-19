"""gamfit vs pyGAM benchmark driver: run a named plan, write records + report.

    python -m bench.pygam_compare.run PLAN --out DIR [--reps R] [--timeout S]
                                              [--memcap-mb M] [--only-libs a,b]
                                              [--shard I/K] [--lib-path DIR]

Writes ``DIR/records.jsonl`` (one JSON object per rep, including reps that
timed out, blew the memory cap, errored or were not run), ``DIR/meta.json``
(host, versions, git sha, pinned thread env, safety-net values) and
``DIR/report.md`` (see ``report.py``).

Every rep is its own subprocess (``worker.py``) so import cost, cold-fit cost
and peak RSS are per-rep, and one library's allocator state or thread pool
never leaks into the next measurement. Within a cell the libraries are
interleaved rep by rep, so slow drift in host load hits all of them alike.

The per-rep timeout and memory cap are a HARNESS SAFETY NET, not a solver
budget: they only stop a runaway rep from stalling the whole plan. A rep that
trips either is recorded with that status, the remaining reps of the cell and
every larger ``n`` of the same (lib, family, design) are recorded as
``not_run_after_<status>``, and the report counts all of it against the
library.

Workers never inherit the caller's ``PYTHONPATH``; ``--lib-path DIR`` is the
one way to put a pinned build first on their import path (for example a copy
of an editable ``gamfit`` taken before a rebuild overwrites it), so a
before/after comparison measures the build it names. Every gamfit record
carries ``lib_file``, the imported package's path, so a run can be audited.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil

from . import report
from .plans import PLANS, Cell, Plan

HERE = Path(__file__).resolve().parent
WORKER = HERE / "worker.py"
REPO_ROOT = HERE.parent.parent
SCHEMA_VERSION = 1
POLL_S = 0.05

# Every BLAS / OpenMP / Rayon pool gets one thread, so the comparison is
# single-core CPU against single-core CPU. pyGAM's scipy/numpy BLAS and
# gamfit's Rayon + faer pools are otherwise sized to the host and a many-core
# runner would measure parallelism, not the algorithms.
THREAD_ENV: dict[str, str] = {
    "RAYON_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _tree(proc: psutil.Process) -> list[psutil.Process]:
    try:
        return [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return [proc]


def _sample(procs: list[psutil.Process]) -> tuple[float, int]:
    rss = 0.0
    threads = 0
    for p in procs:
        try:
            rss += p.memory_info().rss / 2**20
            threads += p.num_threads()
        except psutil.Error:
            continue
    return rss, threads


def run_rep(
    lib: str,
    cell: Cell,
    seed: int,
    timeout_s: float,
    memcap_mb: float,
    cwd: str,
    lib_path: str | None = None,
) -> dict[str, Any]:
    """Run one worker subprocess, policing the safety net; return its record."""
    env = dict(os.environ)
    env.update(THREAD_ENV)
    env.pop("PYTHONPATH", None)
    if lib_path is not None:
        env["PYTHONPATH"] = lib_path
    cmd = [
        sys.executable,
        str(WORKER),
        lib,
        cell.family,
        str(cell.n),
        cell.design,
        str(seed),
    ]
    load_start = os.getloadavg()
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    ps = psutil.Process(proc.pid)
    peak_tree_rss = 0.0
    peak_threads = 0
    status = "ok"
    while proc.poll() is None:
        rss, threads = _sample(_tree(ps))
        peak_tree_rss = max(peak_tree_rss, rss)
        peak_threads = max(peak_threads, threads)
        elapsed = time.perf_counter() - t0
        if rss > memcap_mb:
            status = "memcap"
        elif elapsed > timeout_s:
            status = "timeout"
        if status != "ok":
            for p in reversed(_tree(ps)):
                try:
                    p.kill()
                except psutil.Error:
                    pass
            break
        time.sleep(POLL_S)
    stdout, stderr = proc.communicate()
    wall = time.perf_counter() - t0
    rec: dict[str, Any] = {}
    if status == "ok":
        for line in stdout.splitlines():
            if line.startswith("RESULT "):
                rec = json.loads(line[len("RESULT ") :])
        if not rec:
            status = "crash"
        else:
            status = str(rec.get("status", "error"))
    rec.update(
        lib=lib,
        family=cell.family,
        n=cell.n,
        design=cell.design,
        seed=seed,
        status=status,
        returncode=proc.returncode,
        proc_wall_s=wall,
        peak_tree_rss_mb=peak_tree_rss,
        peak_threads=peak_threads,
        load_start=list(load_start),
        load_end=list(os.getloadavg()),
    )
    if status != "ok":
        rec["stderr_tail"] = stderr[-2000:]
    return rec


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        )
    except OSError:
        return None
    return out.stdout.strip() or None


def run_plan(
    plan: Plan,
    out_dir: Path,
    memcap_mb: float,
    progress: bool = True,
    lib_path: str | None = None,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    meta = {
        "schema_version": SCHEMA_VERSION,
        "plan": dataclasses.asdict(plan),
        "memcap_mb": memcap_mb,
        "safety_net": "timeout_s and memcap_mb are a harness safety net, not a solver budget",
        "thread_env": THREAD_ENV,
        "lib_path": lib_path,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "nproc": os.cpu_count(),
        "total_ram_mb": psutil.virtual_memory().total / 2**20,
        "git_sha": git_sha(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    records: list[dict[str, Any]] = []
    # (lib, family, design) -> status that stopped it at some n
    stopped: dict[tuple[str, str, str], str] = {}
    with (
        tempfile.TemporaryDirectory(prefix="pygam_compare_") as cwd,
        records_path.open("w") as fh,
    ):
        for cell in plan.cells:
            for rep in range(plan.reps):
                for lib in plan.libs:
                    key = (lib, cell.family, cell.design)
                    if key in stopped:
                        rec: dict[str, Any] = dict(
                            lib=lib,
                            family=cell.family,
                            n=cell.n,
                            design=cell.design,
                            seed=rep,
                            status=f"not_run_after_{stopped[key]}",
                        )
                    else:
                        rec = run_rep(
                            lib, cell, rep, plan.timeout_s, memcap_mb, cwd, lib_path
                        )
                        if rec["status"] in ("timeout", "memcap"):
                            stopped[key] = rec["status"]
                    records.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    if progress:
                        print(
                            f"[{cell.key} seed={rep}] {lib:9s} {rec['status']:>8s} "
                            f"fit_cpu={rec.get('fit_cpu_s', float('nan')):.3f}s",
                            file=sys.stderr,
                            flush=True,
                        )
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["lib_versions"] = sorted(
        {f"{r['lib']}={r['lib_version']}" for r in records if r.get("lib_version")}
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out_dir / "report.md").write_text(report.render(records, [meta]))
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("plan", choices=sorted(PLANS), help="named plan (see plans.py)")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--reps", type=int, help="override the plan's reps")
    ap.add_argument(
        "--timeout",
        type=float,
        help="override the per-rep safety-net timeout (seconds)",
    )
    ap.add_argument(
        "--memcap-mb",
        type=float,
        default=psutil.virtual_memory().total / 2**20 / 2,
        help="per-rep process-tree RSS safety net (default: half of total RAM)",
    )
    ap.add_argument("--only-libs", help="comma-separated subset of the plan's libs")
    ap.add_argument(
        "--shard",
        help="I/K: run only the designs whose index in the plan is I mod K, so "
        "K drivers can split a plan across cores (merge with report.py)",
    )
    ap.add_argument(
        "--lib-path",
        type=Path,
        help="directory put first on every worker's PYTHONPATH (a pinned "
        "library build); the caller's own PYTHONPATH is never inherited",
    )
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
    if args.shard:
        index, count = (int(v) for v in args.shard.split("/"))
        if not 0 <= index < count:
            ap.error(f"--shard {args.shard}: need 0 <= I < K")
        # Shard by design, so every n of one design stays in one driver and
        # the not-run-after-timeout rule still sees it.
        designs = list(dict.fromkeys(c.design for c in plan.cells))
        mine = {d for j, d in enumerate(designs) if j % count == index}
        plan = dataclasses.replace(
            plan, cells=tuple(c for c in plan.cells if c.design in mine)
        )
    lib_path = None
    if args.lib_path is not None:
        if not args.lib_path.is_dir():
            ap.error(f"--lib-path {args.lib_path}: not a directory")
        lib_path = str(args.lib_path.resolve())
    records = run_plan(
        plan, args.out, args.memcap_mb, progress=not args.quiet, lib_path=lib_path
    )
    print(f"wrote {len(records)} records to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
