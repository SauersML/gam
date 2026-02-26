#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN_SUITE = ROOT / "bench" / "run_suite.py"
SCENARIOS = ROOT / "bench" / "scenarios.json"


def run_cmd(cmd):
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
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
        if metric is None:
            metric = r.get("auc")
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
