#!/usr/bin/env python3
"""Run every original #2668 contract; missing tests and timeouts are not passes."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        parser.error("workers and timeout must be positive")
    binary = args.binary.resolve(strict=True)
    root = Path(__file__).resolve().parent.parent
    entries = json.loads((root / "tests/data/issue_2668_regressions.json").read_text())
    inventory = subprocess.run(
        [str(binary), "--list", "--format", "terse"],
        check=True, capture_output=True, text=True,
    ).stdout
    names = [line.removesuffix(": test") for line in inventory.splitlines()
             if line.endswith(": test")]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "inventory.txt").write_text(inventory)
    scratch = args.output.resolve().with_name(args.output.name + "-scratch")
    scratch.mkdir(exist_ok=True)
    with binary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    environment = dict(os.environ, TMPDIR=str(scratch), RAYON_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

    def run(entry):
        matches = [name for name in names if name.split("::")[-1] == entry["test"]]
        record = dict(entry, matches=matches)
        if len(matches) != 1:
            return dict(record, status="missing" if not matches else "ambiguous")
        log = args.output / (entry["test"] + ".log")
        start = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.Popen(
                [str(binary), "--exact", matches[0], "--nocapture", "--test-threads=1"],
                cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=args.timeout)
                status = "passed" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                # This process group belongs exclusively to the test we started.
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
                status = "timeout"
        if status == "passed" and "1 passed; 0 failed; 0 ignored;" not in log.read_text():
            status = "unmeasured"
        record.update(status=status, exit_code=code,
                      seconds=time.monotonic() - start, log=str(log))
        return record

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run, entry) for entry in entries]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"{record['status']:10} {record['original']}", flush=True)
            receipt = dict(binary=str(binary), binary_sha256=digest,
                           test_timeout_seconds=args.timeout, workers=args.workers,
                           results=records)
            (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    counts = {status: sum(r["status"] == status for r in records)
              for status in sorted({r["status"] for r in records})}
    with binary.open("rb") as stream:
        final_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt["binary_sha256_after"] = final_digest
    receipt["binary_unchanged"] = final_digest == digest
    (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(counts, sort_keys=True), flush=True)
    return 0 if final_digest == digest and len(records) == 30 and counts == {"passed": 30} else 1


if __name__ == "__main__":
    raise SystemExit(main())
