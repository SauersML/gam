"""Convergence-fuzz driver: run a plan of (case, family, n) reps, write records.

    cd bench && python -m convergence_fuzz.run PLAN --out DIR [--jobs J]

Plans:

``full``
    cases ``0..FULL_CASES-1`` x every ``n`` of ``N_GRID`` below
    ``LARGE_N`` x every family, plus cases ``0..LARGE_N_CASES-1`` at
    ``LARGE_N``: 1692 reps, 3384 fits (each rep fits the model and its
    permuted/rescaled twin).
``quick``
    the seeded fixture of every root cause this fuzzer found and fixed; the
    regression test (``test_quick.py``) requires zero failures on it. Causes
    still open are tracked by the ``full`` plan's report, not by ``quick``.

Every rep is one ``worker.py`` subprocess launched through
``pygam_compare.run.run_isolated``, the gamfit-vs-pyGAM harness's isolation:
pinned single-thread BLAS/OpenMP/Rayon pools, process-tree RSS polling, and a
scratch working directory. ``--jobs`` runs that many reps side by side, each
single-threaded. The per-rep timeout and memory cap are the harness's safety
net, not a solver budget: a rep that trips either is recorded as a ``hang`` /
``memcap`` failure (see ``triage.py``), never retried or excused.

Writes ``DIR/records.jsonl`` (one object per rep, in plan order),
``DIR/meta.json`` and ``DIR/report.md`` (failure counts by cause).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
from pygam_compare.run import THREAD_ENV, git_sha, run_isolated

from convergence_fuzz import dgp, triage

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
SCHEMA_VERSION = 1

N_GRID: tuple[int, ...] = (30, 100, 1_000, 10_000)
FULL_CASES = 180
# A rep at n = 10 000 costs one to fifteen single-threaded minutes, a hundred
# times one at n = 1 000, so the largest n is drawn on fewer cases. It still
# spans every covariate count (cases 0..23 draw p = 1..8).
LARGE_N = 10_000
LARGE_N_CASES = 24
# Safety net only (see module docstring). The slowest certified reps of the
# full plan (binomial, n = 10 000, both fits) take about five minutes
# single-threaded under full-host load.
TIMEOUT_S = 900.0


@dataclass(frozen=True)
class Rep:
    case: int
    family: str
    n: int

    @property
    def key(self) -> str:
        return f"case{self.case}/{self.family}/n{self.n}"


# One seeded rep per root cause this fuzzer found and fixed in the engine: the
# smallest failing (case, family, n) of that cause's cluster in the "before"
# run. Kept in ``quick`` forever, so the cause cannot come back unnoticed.
FIXTURES: dict[str, Rep] = {
    # The latched #784 block was re-ranked by |gamma| at every rho, so the
    # spliced criterion jumped where two directions' |gamma| crossed.
    "block-correction-reselected": Rep(0, "binomial", 1000),
}


def _full() -> list[Rep]:
    return [
        Rep(case, family, n)
        for case in range(FULL_CASES)
        for n in N_GRID
        if n < LARGE_N or case < LARGE_N_CASES
        for family in dgp.FAMILIES
    ]


def _quick() -> list[Rep]:
    return list(dict.fromkeys(FIXTURES.values()))


PLANS = {"full": _full, "quick": _quick}


def run_one(rep: Rep, cwd: str, timeout_s: float, memcap_mb: float) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "convergence_fuzz.worker",
        str(rep.case),
        rep.family,
        str(rep.n),
    ]
    rec = run_isolated(
        cmd, cwd, timeout_s, memcap_mb, env_extra={"PYTHONPATH": str(BENCH_DIR)}
    )
    rec.update(case=rep.case, family=rep.family, n=rep.n)
    rec["causes"] = triage.failure_causes(rec)
    return rec


def run_plan(
    reps: list[Rep],
    out_dir: Path,
    jobs: int,
    timeout_s: float,
    memcap_mb: float,
    plan_name: str,
    progress: bool = True,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "plan": plan_name,
        "reps": len(reps),
        "jobs": jobs,
        "timeout_s": timeout_s,
        "memcap_mb": memcap_mb,
        "safety_net": "timeout_s and memcap_mb are a harness safety net, not a solver budget",
        "thread_env": THREAD_ENV,
        "root_seed": dgp.ROOT_SEED,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "nproc": os.cpu_count(),
        "git_sha": git_sha(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    records: list[dict[str, Any]] = []
    with (
        tempfile.TemporaryDirectory(prefix="convergence_fuzz_") as cwd,
        (out_dir / "records.jsonl").open("w") as fh,
        ThreadPoolExecutor(max_workers=jobs) as pool,
    ):
        futures = [pool.submit(run_one, rep, cwd, timeout_s, memcap_mb) for rep in reps]
        for i, (rep, fut) in enumerate(zip(reps, futures)):
            rec = fut.result()
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if progress:
                verdict = ",".join(rec["causes"]) or "ok"
                print(
                    f"[{i + 1}/{len(reps)}] {rep.key:28s} "
                    f"{rec['proc_wall_s']:7.2f}s  {verdict[:160]}",
                    file=sys.stderr,
                    flush=True,
                )
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["lib_versions"] = sorted(
        {str(r["lib_version"]) for r in records if r.get("lib_version")}
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out_dir / "report.md").write_text(triage.render(records))
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("plan", choices=sorted(PLANS))
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="reps run side by side, each single-threaded (default: nproc)",
    )
    ap.add_argument(
        "--memcap-mb",
        type=float,
        help="per-rep process-tree RSS safety net (default: total RAM / (2 jobs))",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    memcap = args.memcap_mb or psutil.virtual_memory().total / 2**20 / (2 * args.jobs)
    records = run_plan(
        PLANS[args.plan](),
        args.out,
        args.jobs,
        TIMEOUT_S,
        memcap,
        args.plan,
        progress=not args.quiet,
    )
    failed = sum(1 for r in records if r["causes"])
    print(
        f"wrote {len(records)} records to {args.out}; {failed} with a failure",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
