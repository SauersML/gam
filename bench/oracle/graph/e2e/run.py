"""End-to-end run of the graph oracle's path (#2951): behavior file -> oracle prompt (prompt.py) ->
program -> mech.trace IR -> Rust checker (score.py) -> reader term (reader_score.py server at
GRAPH_READER, when set) -> every score term. Appends one line per program to the status table
~/mpd-data/graph_oracle/runs/status.tsv.

  run.py BEHAVIOR.json [--programs hand empty random full | FILE.py ...] [--experiments 32] [--seed 0]

Programs named hand/empty/random/full come from programs.py; anything else is read as a source file.
The checker binary is GRAPH_CHECKER (score.py's default otherwise).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
# The tracer's child is a venv script: without this it reserves the venv default (4 GiB of memory
# ledger tokens) and waits for them, which turns a 0.2 s trace into a timeout.
os.environ.setdefault("MPD_MEM_GIB", "1")

import programs  # noqa: E402
import prompt  # noqa: E402
import score  # noqa: E402

STATUS = Path.home() / "mpd-data/graph_oracle/runs/status.tsv"
COLUMNS = ["time", "model", "behavior", "program", "total_bits", "exec_error_bits", "reader_error_bits", "code_bits",
           "python_tokens", "opaque_numbers", "opaque_bits", "N", "experiments", "valid", "clean_kl_bits", "checker"]


def sources(names: list[str], model: str, seed: int) -> dict[str, str]:
    refs = programs.references(model, seed)
    return {Path(n).stem if n not in refs else n: refs[n] if n in refs else Path(n).read_text() for n in names}


def status_line(model: str, behavior: str, name: str, result: dict) -> list[str]:
    clean = result.get("per_family", {}).get("clean", {}).get("mean_kl_bits")
    row = {"time": time.strftime("%Y-%m-%d %H:%M"), "model": model, "behavior": behavior, "program": name,
           "clean_kl_bits": clean, "checker": Path(str(score.BINARY)).name, **result}
    return [("" if row.get(c) is None else f"{row[c]:.6g}" if isinstance(row[c], float) else str(row[c])) for c in COLUMNS]


def record(lines: list[list[str]], path: Path = STATUS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a") as f:
        if new:
            f.write("\t".join(COLUMNS) + "\n")
        for line in lines:
            f.write("\t".join(line) + "\n")


def run(behavior_path: Path, names: list[str], experiments: int = 32, seed: int = 0, reader: bool = True,
        write_status: bool = True) -> dict[str, dict]:
    behavior = json.loads(behavior_path.read_text())
    model = behavior["model"]
    text = prompt.render(behavior)  # the oracle's input; the reference programs do not read it
    results = {}
    with score.Checker(model) as checker:
        checker.behavior(behavior_path)
        for name, source in sources(names, model, seed).items():
            t = time.time()
            result = checker.score(source, experiments=experiments, seed=seed, reader=reader)
            result.pop("items", None)
            result["seconds"] = time.time() - t
            results[name] = result
    if write_status:
        record([status_line(model, behavior["id"], n, r) for n, r in results.items()])
    results["_prompt_characters"] = len(text)
    return results


def table(results: dict[str, dict]) -> str:
    keys = ["total_bits", "exec_error_bits", "reader_error_bits", "code_bits", "opaque_numbers", "opaque_bits", "experiments", "valid"]
    out = ["program\t" + "\t".join(keys) + "\tclean_kl_bits"]
    for name, r in results.items():
        if name.startswith("_"):
            continue
        clean = r.get("per_family", {}).get("clean", {}).get("mean_kl_bits")
        cells = ["" if r.get(k) is None else f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k]) for k in keys]
        out.append("\t".join([name] + cells + ["" if clean is None else f"{clean:.4g}"]))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behavior", type=Path)
    ap.add_argument("--programs", nargs="+", default=["hand", "empty", "random", "full"])
    ap.add_argument("--experiments", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-reader", action="store_true")
    ap.add_argument("--json", type=Path, help="also write the full results here")
    a = ap.parse_args()
    results = run(a.behavior.expanduser(), a.programs, a.experiments, a.seed, reader=not a.no_reader)
    print(table(results))
    if a.json:
        a.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
