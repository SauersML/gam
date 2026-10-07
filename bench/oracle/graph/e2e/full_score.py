"""Joins the reader term to the checker's terms (#2951): for each program that run.py scored with
--items, a stratified subsample of its reader items (per experiment family, fixed seed) goes to one
reader_score.py server, and the program's total gains reader_error_bits = N x the subsample's mean bits
per item, each family's sampled mean weighted by the family's share of all items (an unbiased estimate of
the full items' mean). Writes the joined results and status lines.

  full_score.py RESULTS.json ITEMS_DIR BEHAVIOR_ID [--per-program 64] [--model Qwen/Qwen3-8B] [--device mps]
                [--target vpd4l] [--out JOINED.json]
RESULTS.json is run.py --json's output; ITEMS_DIR holds <behavior>.<program>.{items,program}.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run as e2e  # noqa: E402

READER = HERE.parent / "reader_score.py"


def subsample(items: list[dict], k: int, seed: int = 0) -> list[dict]:
    """k items, spread over the experiment families as evenly as their counts allow."""
    if len(items) <= k:
        return items
    rng = random.Random(seed)
    by = defaultdict(list)
    for it in items:
        by[it.get("family", "all")].append(it)
    for v in by.values():
        rng.shuffle(v)
    out, families = [], sorted(by)
    while len(out) < k:
        for f in families:
            if by[f] and len(out) < k:
                out.append(by[f].pop())
    return out


def ask(port: int, message: dict) -> dict:
    with socket.create_connection(("127.0.0.1", port)) as s:
        s.sendall((json.dumps(message) + "\n").encode())
        reply = json.loads(s.makefile().readline())
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply["ok"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("items", type=Path)
    ap.add_argument("behavior")
    ap.add_argument("--per-program", type=int, default=64)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default=None)
    ap.add_argument("--target", default="vpd4l")
    ap.add_argument("--port", type=int, default=47000 + os.getpid() % 1000)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    results = json.loads(a.results.read_text())
    command = [sys.executable, str(READER), "serve", "--model", a.model, "--target", a.target, "--listen", f"127.0.0.1:{a.port}"]
    if a.device:
        command += ["--device", a.device]
    server = subprocess.Popen(command, stderr=open(a.results.with_suffix(".reader.log"), "w"))
    try:
        for _ in range(1800):
            try:
                socket.create_connection(("127.0.0.1", a.port)).close()
                break
            except OSError:
                if server.poll() is not None:
                    sys.exit(f"the reader server exited ({a.results.with_suffix('.reader.log')})")
                time.sleep(1)
        lines = []
        for name, r in results.items():
            stem = a.items / f"{a.behavior}.{name}"
            if name.startswith("_") or not Path(f"{stem}.items.jsonl").exists():
                continue
            items = [json.loads(l) for l in Path(f"{stem}.items.jsonl").read_text().splitlines() if l.strip()]
            program = json.loads(Path(f"{stem}.program.jsonl").read_text())
            t = time.time()
            sample = subsample(items, a.per_program)
            reply = ask(a.port, {"op": "score", "programs": [program], "items": sample, "N": int(r["N"])})
            reader = reply["results"][0]
            share = defaultdict(int)
            for it in items:
                share[it.get("family", "all")] += 1
            sampled = defaultdict(list)
            for it, bits in zip(sample, reader["per_item"]):
                sampled[it.get("family", "all")].append(bits)
            mean = sum(share[f] / len(items) * sum(v) / len(v) for f, v in sampled.items())
            reader["reader_error_bits"] = r["N"] * mean
            r["reader"] = {k: v for k, v in reader.items() if k != "per_item"} | {"items_scored": min(len(items), a.per_program),
                                                                               "items_total": len(items), "seconds": time.time() - t,
                                                                               "model": a.model}
            r["reader_error_bits"] = reader["reader_error_bits"]
            r["total_bits"] = r["exec_error_bits"] + r["code_bits"] + r["opaque_bits"] + r["reader_error_bits"]
            lines.append(e2e.status_line(a.target, a.behavior, f"{name}+reader:{a.model.split('/')[-1]}", r, r.get("stand_in")))
            print(name, {k: round(r[k] / r["N"], 4) for k in ("total_bits", "exec_error_bits", "opaque_bits", "code_bits", "reader_error_bits")},
                  "english saved/N", round(reader.get("english_saved_bits", float("nan")) / r["N"], 4), flush=True)
        e2e.record(lines)
        (a.out or a.results.with_suffix(".joined.json")).write_text(json.dumps(results, indent=1))
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
