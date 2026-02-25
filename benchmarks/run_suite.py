#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
DEFAULT_SCENARIOS = BENCH_DIR / "scenarios.json"


def run_cmd(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def run_rust_scenario(s):
    cmd = [
        "cargo",
        "run",
        "--release",
        "--example",
        "bench_fit",
        "--",
        "--n",
        str(s["n"]),
        "--p",
        str(s["p"]),
        "--seed",
        str(s.get("seed", 42)),
    ]
    code, out, err = run_cmd(cmd, cwd=ROOT)
    if code != 0:
        return {
            "contender": "rust_gam",
            "scenario": s["name"],
            "status": "failed",
            "error": err.strip() or out.strip(),
        }
    line = out.strip().splitlines()[-1]
    row = json.loads(line)
    row["contender"] = "rust_gam"
    row["scenario_name"] = s["name"]
    row["status"] = "ok"
    return row


def run_external_placeholder(contender, scenario):
    return {
        "contender": contender,
        "scenario": scenario["name"],
        "status": "skipped",
        "reason": "Adapter hook only. Wire your local command in benchmarks/README.md.",
    }


def main():
    parser = argparse.ArgumentParser(description="Run GAM benchmark suite.")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--out", type=Path, default=BENCH_DIR / "results.json")
    parser.add_argument(
        "--with-external",
        action="store_true",
        help="Include placeholder entries for R mgcv / Python pygam contenders.",
    )
    args = parser.parse_args()

    cfg = json.loads(args.scenarios.read_text())
    scenarios = cfg.get("scenarios", [])
    if not scenarios:
        print("No scenarios found.", file=sys.stderr)
        sys.exit(1)

    results = []
    for s in scenarios:
        results.append(run_rust_scenario(s))
        if args.with_external:
            results.append(run_external_placeholder("r_mgcv", s))
            results.append(run_external_placeholder("python_pygam", s))

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

