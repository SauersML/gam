#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_SUITE = ROOT / "bench" / "run_suite.py"
SCENARIOS = ROOT / "benchmarks" / "scenarios.json"
DEFAULT_SURVIVAL_SCENARIOS = [
    "heart_failure_survival",
    "icu_survival_death",
    "icu_survival_los",
    "cirrhosis_survival",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run survival benchmarks with leakage-safe 5-fold CV only."
    )
    parser.add_argument(
        "--scenario-name",
        action="append",
        dest="scenario_names",
        help="Survival scenario(s) to run. Defaults to all survival scenarios.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmarks" / "results.survival.json",
    )
    args = parser.parse_args()

    scenarios = args.scenario_names or DEFAULT_SURVIVAL_SCENARIOS
    cmd = [
        sys.executable,
        str(RUN_SUITE),
        "--scenarios",
        str(SCENARIOS),
        "--out",
        str(args.out),
    ]
    for scenario in scenarios:
        cmd.extend(["--scenario-name", scenario])

    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print((proc.stderr or proc.stdout).strip(), file=sys.stderr)
        return proc.returncode

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
