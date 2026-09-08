"""Bounded MSI benchmark of the production timewiggle source using warm rlibs.

Pass a saved successful rustc argv JSON as --dependencies. This deliberately
does not invoke Cargo or rebuild dependencies. Run on MSI, not locally.
"""

import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--cpus", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    source_argv = json.loads(args.dependencies.read_text())
    externs = {}
    dependency_search = []
    for index, item in enumerate(source_argv[:-1]):
        if item == "--extern":
            name, artifact = source_argv[index + 1].split("=", 1)
            externs[name] = artifact
        elif item == "-L" and source_argv[index + 1].startswith("dependency="):
            dependency_search.extend([item, source_argv[index + 1]])
    binary = root / ".buildd/timewiggle_q_932"
    command = [
        source_argv[0], "--crate-name", "timewiggle_q_932", "--edition=2024",
        "--test", "bench/measurements/issue_932/timewiggle_q.rs", "--deny=warnings",
        "-Copt-level=3", "-Ccodegen-units=1", "-Clto=off", "-o", str(binary),
        *dependency_search,
    ]
    for name in ("gam_math", "ndarray"):
        command.extend(["--extern", f"{name}={externs[name]}"])
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.with_suffix(".argv.json").write_text(json.dumps(command, indent=2) + "\n")
    start = time.monotonic()
    with args.log.open("w") as log:
        built = subprocess.run(
            ["taskset", "-c", args.cpus, *command], cwd=root,
            stdout=log, stderr=subprocess.STDOUT, timeout=45,
        )
        print(f"build={built.returncode} seconds={time.monotonic() - start:.2f}", flush=True)
        if built.returncode:
            status = built.returncode
        else:
            tested = subprocess.run(
                ["taskset", "-c", args.cpus, str(binary), "--nocapture", "--test-threads=1"],
                cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=30,
            )
            status = tested.returncode
    print(args.log.read_text())
    raise SystemExit(status)


if __name__ == "__main__":
    main()
