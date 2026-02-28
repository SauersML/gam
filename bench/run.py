#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN_SUITE = ROOT / "bench" / "run_suite.py"
SCENARIOS = ROOT / "bench" / "scenarios.json"
SERIAL_ENV_OVERRIDES = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "CARGO_BUILD_JOBS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
}


def run_cmd(cmd):
    env = os.environ.copy()
    env.update(SERIAL_ENV_OVERRIDES)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    p = argparse.ArgumentParser(
    )
    p.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Scenario name(s). Omit to run all.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output JSON path. Default: temp file.",
    )
    args = p.parse_args()

    out_path = args.out
    cleanup = False
    if out_path is None:
        td = tempfile.TemporaryDirectory(prefix="gam_magic_bench_")
        cleanup = True
        out_path = Path(td.name) / "results.json"

    cmd = [sys.executable, str(RUN_SUITE), "--scenarios", str(SCENARIOS), "--out", str(out_path)]
    for s in args.scenarios or []:
        cmd.extend(["--scenario-name", s])

    code, out, err = run_cmd(cmd)
    if code != 0:
        print(err.strip() or out.strip(), file=sys.stderr)
        return code

    payload = json.loads(out_path.read_text())
    rows = payload.get("results", [])

    print("-" * 78)
    for r in rows:
        metric = r.get("auc")
        if metric is None:
            metric = r.get("c_index")
        print(
            f"{r.get('contender','-')} | "
            f"{r.get('scenario_name','-')} | "
            f"{r.get('status','-')} | "
            f"{fmt(metric)} | "
            f"{fmt(r.get('brier'))} | "
            f"{fmt(r.get('rmse'))} | "
            f"{fmt(r.get('r2'))}"
        )

    if args.out is not None:
        print(f"\nWrote: {out_path}")

    if cleanup:
        td.cleanup()  # type: ignore[name-defined]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
